from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ArmType:
    name: str
    count: int
    p: float
    w: float
    initial_state: int


ARM_TYPES = (
    ArmType("Type 1", 25, 0.95, 0.90, 1),
    ArmType("Type 2", 25, 0.95, 0.20, 2),
    ArmType("Type 3", 25, 0.70, 0.95, 2),
    ArmType("Type 4", 25, 0.70, 0.20, 2),
)

N_ARMS = 100
BUDGET = 16


def cycle_gain(threshold: int, lam: float, p: float, w: float) -> float:
    """Average reward for a threshold policy in the restart model."""

    if threshold < 1:
        raise ValueError("threshold must be >= 1")

    cycle_len = threshold - 1 + 1.0 / p
    passive_age_sum = (threshold - 1) * threshold / 2.0
    active_age_sum = threshold / p + (1.0 - p) / (p * p)
    active_count = 1.0 / p
    cycle_reward = -w * (passive_age_sum + active_age_sum) + lam * active_count
    return cycle_reward / cycle_len


def continuous_threshold(lam: float, p: float, w: float) -> float:
    radicand = (1.0 - p) ** 2 - 2.0 * p * lam / w
    if radicand <= 0.0:
        return 1.0
    return max(1.0, (math.sqrt(radicand) - (1.0 - p)) / p)


def optimal_threshold_and_gain(lam: float, p: float, w: float) -> tuple[int, float]:
    x_cont = continuous_threshold(lam, p, w)
    candidates = {
        1,
        max(1, math.floor(x_cont)),
        max(1, math.ceil(x_cont)),
        max(1, math.floor(x_cont) - 1),
        max(1, math.ceil(x_cont) + 1),
    }
    best_x = 1
    best_g = -math.inf
    for x in sorted(candidates):
        g = cycle_gain(x, lam, p, w)
        if g > best_g:
            best_x = x
            best_g = g
    return best_x, best_g


def lagrange_value(lam: float, arm_types: tuple[ArmType, ...] = ARM_TYPES) -> float:
    gains = 0.0
    for arm_type in arm_types:
        _, gain = optimal_threshold_and_gain(lam, arm_type.p, arm_type.w)
        gains += arm_type.count * gain
    return gains - lam * BUDGET


def value_function_at(state: int, threshold: int, gain: float, lam: float, p: float, w: float) -> float:
    """Relative value V(x), with V(1)=0, for the optimal threshold at lambda."""

    if state <= 1:
        return 0.0

    if state < threshold:
        return gain * (state - 1) + w * (state - 1) * state / 2.0

    value = gain * (threshold - 1) + w * (threshold - 1) * threshold / 2.0
    for x in range(threshold, state):
        value = (value + w * x - lam + gain) / (1.0 - p)
    return value


def lagrangian_index(state: int, lam: float, p: float, w: float) -> float:
    threshold, gain = optimal_threshold_and_gain(lam, p, w)
    return lam - p * value_function_at(state + 1, threshold, gain, lam, p, w)


def build_arm_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    ps: list[float] = []
    ws: list[float] = []
    states: list[int] = []
    type_ids: list[int] = []
    labels: list[str] = []
    for type_id, arm_type in enumerate(ARM_TYPES):
        ps.extend([arm_type.p] * arm_type.count)
        ws.extend([arm_type.w] * arm_type.count)
        states.extend([arm_type.initial_state] * arm_type.count)
        type_ids.extend([type_id] * arm_type.count)
        labels.extend([arm_type.name] * arm_type.count)
    return np.array(ps), np.array(ws), np.array(states, dtype=int), np.array(type_ids, dtype=int), labels


class RestartIndexCache:
    def __init__(self, lam: float, arm_type: ArmType) -> None:
        self.lam = lam
        self.p = arm_type.p
        self.w = arm_type.w
        self.threshold, self.gain = optimal_threshold_and_gain(lam, self.p, self.w)
        self.values = [0.0, 0.0]
        self.indices = [0.0]

    def _value_formula_before_threshold(self, state: int) -> float:
        return self.gain * (state - 1) + self.w * (state - 1) * state / 2.0

    def ensure_values(self, state: int) -> None:
        """Ensure V(state) is available."""

        target_value_state = state + 1
        while len(self.values) <= target_value_state:
            x = len(self.values)
            if x < self.threshold:
                value = self._value_formula_before_threshold(x)
            elif x == self.threshold:
                value = self._value_formula_before_threshold(x)
            else:
                prev_x = x - 1
                prev_value = self.values[-1]
                value = (prev_value + self.w * prev_x - self.lam + self.gain) / (1.0 - self.p)
            self.values.append(value)

    def ensure_state(self, state: int) -> None:
        self.ensure_values(state)
        while len(self.indices) <= state:
            x = len(self.indices)
            self.ensure_values(x)
            self.indices.append(self.lam - self.p * self.values[x + 1])

    def get_many(self, states: np.ndarray) -> np.ndarray:
        max_state = int(states.max())
        self.ensure_state(max_state)
        return np.take(np.asarray(self.indices), states)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="same")


def plot_figure_1b(args: argparse.Namespace, output_dir: Path) -> dict[str, float]:
    coarse_lambdas = np.linspace(args.lambda_min, args.lambda_max, args.grid_size)
    coarse_values = np.array([lagrange_value(float(lam)) for lam in coarse_lambdas])
    coarse_best = float(coarse_lambdas[int(np.argmin(coarse_values))])

    refine_lambdas = np.linspace(
        coarse_best - args.refine_width,
        coarse_best + args.refine_width,
        args.refine_grid_size,
    )
    refine_values = np.array([lagrange_value(float(lam)) for lam in refine_lambdas])
    best_idx = int(np.argmin(refine_values))
    best_lambda = float(refine_lambdas[best_idx])
    best_value = float(refine_values[best_idx])

    display_mask = (coarse_lambdas >= args.display_min) & (coarse_lambdas <= args.display_max)
    fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=160)
    ax.plot(coarse_lambdas[display_mask], coarse_values[display_mask], color="#1f77b4", linewidth=1.8)
    ax.axvline(best_lambda, color="#d62728", linewidth=1.5, label=fr"$\lambda^*={best_lambda:.4f}$")
    ax.scatter([best_lambda], [best_value], color="#d62728", s=22, zorder=3)
    ax.set_xlim(args.display_min, args.display_max)
    ax.set_xlabel(r"Lagrange multiplier $\lambda$")
    ax.set_ylabel(r"Optimal Lagrange function")
    ax.set_title("Restart Model: Optimal Lagrange Function")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    path = output_dir / "figure_1b_lagrange_function.png"
    fig.savefig(path)
    plt.close(fig)

    csv_path = output_dir / "figure_1b_curve.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lambda", "lagrange_value"])
        writer.writerows(zip(coarse_lambdas, coarse_values))

    return {
        "figure_1b_lambda_star": best_lambda,
        "figure_1b_value_star": best_value,
    }


def simulate_lip(args: argparse.Namespace, output_dir: Path) -> dict[str, float]:
    ps, ws, initial_states, type_ids, labels = build_arm_arrays()
    index_caches = [RestartIndexCache(args.policy_lambda, arm_type) for arm_type in ARM_TYPES]
    rng = np.random.default_rng(args.seed)
    reward_paths = np.zeros((args.mc_times, args.horizon), dtype=float)
    activation_counts = {arm_type.name: 0 for arm_type in ARM_TYPES}

    for run in range(args.mc_times):
        states = initial_states.copy()
        for t in range(args.horizon):
            indices = np.empty(N_ARMS, dtype=float)
            for type_id in range(len(ARM_TYPES)):
                mask = type_ids == type_id
                indices[mask] = index_caches[type_id].get_many(states[mask])
            active = np.argpartition(indices, -BUDGET)[-BUDGET:]
            active_set = set(int(i) for i in active)

            reward_paths[run, t] = float(np.sum(-ws * states))
            successes = rng.random(BUDGET) < ps[active]
            next_states = states + 1
            next_states[active[successes]] = 1
            states = next_states

            if run == 0:
                for i in active_set:
                    activation_counts[labels[i]] += 1

    mean_reward = reward_paths.mean(axis=0)
    moving_reward = rolling_mean(mean_reward, args.reward_window)

    fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=160)
    ax.plot(np.arange(1, args.horizon + 1), moving_reward, color="#2ca02c", linewidth=1.5, label="LIP")
    ax.set_xlabel("Time")
    ax.set_ylabel("Average reward")
    ax.set_title("Restart Model: LIP Average Reward")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    path = output_dir / "figure_2_lip_average_reward.png"
    fig.savefig(path)
    plt.close(fig)

    csv_path = output_dir / "figure_2_lip_reward.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "mc_mean_reward", f"moving_mean_window_{args.reward_window}"])
        for t, (raw, smooth) in enumerate(zip(mean_reward, moving_reward), start=1):
            writer.writerow([t, raw, smooth])

    type_activation_rates = {
        key: value / (args.horizon * BUDGET) for key, value in activation_counts.items()
    }
    return {
        "figure_2_final_moving_reward": float(moving_reward[-1]),
        "figure_2_mean_reward_last_1000": float(mean_reward[-1000:].mean()),
        **{f"run0_activation_share_{key.replace(' ', '_').lower()}": float(val) for key, val in type_activation_rates.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce restart-model Figure 1(b) and Figure 2.")
    parser.add_argument("--lambda-min", type=float, default=-30.0)
    parser.add_argument("--lambda-max", type=float, default=5.0)
    parser.add_argument("--grid-size", type=int, default=1401)
    parser.add_argument("--refine-width", type=float, default=0.05)
    parser.add_argument("--refine-grid-size", type=int, default=401)
    parser.add_argument("--display-min", type=float, default=-20.0)
    parser.add_argument("--display-max", type=float, default=0.0)
    parser.add_argument("--policy-lambda", type=float, default=-11.6400)
    parser.add_argument("--horizon", type=int, default=10000)
    parser.add_argument("--mc-times", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--reward-window", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("output/restart_figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    summary.update(plot_figure_1b(args, args.output_dir))
    summary.update(simulate_lip(args, args.output_dir))

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
