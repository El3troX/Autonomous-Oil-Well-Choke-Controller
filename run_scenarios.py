"""
Closed-Loop Scenario Runner
============================
Runs the MPC controller against the TRUE simulator (not the identified
model) for three demonstration scenarios:

  A — Startup to Target: idle → target production rate
  B — Target Tracking: rate changes mid-run (100 → 150 bbl/hr)
  C — Infeasible Target: target exceeds safe maximum → settle at max

Each scenario produces:
  - A CSV log of every control step
  - A 6-panel plot (Q, WHP, FLP, BHP, Choke vs time)
  - A pass/fail summary printed to console
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulator import OilWellSimulator
from well_model import WellModel
from controller import MPCController
from config import (
    WHP_MIN, WHP_MAX, FLP_MIN, FLP_MAX, BHP_MIN, BHP_MAX,
    CHOKE_RAMP_MAX,
)
from paths import data_path, image_path


def run_scenario(name, sim, controller, target_schedule, start_choke,
                 n_steps):
    """
    Run a single closed-loop scenario.

    Parameters
    ----------
    name : str
        Scenario label (for filenames/plot titles).
    sim : OilWellSimulator
        The TRUE simulator (reality).
    controller : MPCController
        The MPC controller (uses identified model internally).
    target_schedule : list of (start_step, target_Q)
        Time-varying target oil rate schedule.
    start_choke : float
        Initial choke position (%).
    n_steps : int
        Total number of control steps to run.
    Returns
    -------
    df : pd.DataFrame
        Logged step-by-step data.
    summary : dict
        Pass/fail summary metrics.
    """
    # Reset simulator to initial state
    sim.reset(choke_position=start_choke)
    Q, WHP, FLP, BHP = sim.step(start_choke)

    # Build target lookup
    def get_target(step):
        target = target_schedule[0][1]
        for t_start, t_val in sorted(target_schedule, key=lambda item: item[0]):
            if step >= t_start:
                target = t_val
        return target

    # Run closed loop
    records = []
    current_choke = start_choke

    for step in range(n_steps):
        target_Q = get_target(step)

        state = {
            "Q": Q, "WHP": WHP, "FLP": FLP, "BHP": BHP,
            "choke": current_choke,
        }
        next_choke, info = controller.compute_action(state, target_Q)
        Q, WHP, FLP, BHP = sim.step(next_choke)

        records.append({
            "Time_hr": step + 1,
            "Choke_pct": next_choke,
            "Target_Q": target_Q,
            "Actual_Q": Q,
            "WHP_psi": WHP,
            "FLP_psi": FLP,
            "BHP_psi": BHP,
            "Feasible_Candidates": info["feasible_candidates"],
        })

        controller.log_step(step + 1, next_choke, target_Q, Q, WHP, FLP, BHP, info)
        current_choke = next_choke

    df = pd.DataFrame(records)

    # ── Summary checks ─────────────────────────────────────────────
    # Check ramp rate compliance
    choke_diffs = df["Choke_pct"].diff().abs().fillna(0)
    ramp_violations = (choke_diffs > CHOKE_RAMP_MAX + 0.01).sum()

    # Check safety constraint violations
    whp_violations = ((df["WHP_psi"] < WHP_MIN) | (df["WHP_psi"] > WHP_MAX)).sum()
    flp_violations = ((df["FLP_psi"] < FLP_MIN) | (df["FLP_psi"] > FLP_MAX)).sum()
    bhp_violations = ((df["BHP_psi"] < BHP_MIN) | (df["BHP_psi"] > BHP_MAX)).sum()
    any_safety_violation = (whp_violations + flp_violations + bhp_violations) > 0

    # Check target achievement (use last 20% of data as "settled" region)
    n_settle = max(5, int(n_steps * 0.2))
    settled = df.tail(n_settle)
    final_target = df["Target_Q"].iloc[-1]
    avg_settled_Q = settled["Actual_Q"].mean()
    max_settled_Q = settled["Actual_Q"].max()
    min_settled_Q = settled["Actual_Q"].min()
    # Derive the theoretical max from the simulator instead of relying on
    # a hard-coded value. This keeps the scenario check aligned with the
    # process model if the calibration ever changes.
    max_achievable = float(sim._steady_state(100.0)[0])
    target_is_infeasible = final_target > max_achievable + 10
    if target_is_infeasible:
        target_achieved = abs(avg_settled_Q - max_achievable) < 15
    else:
        target_achieved = abs(avg_settled_Q - final_target) < 10

    summary = {
        "scenario": name,
        "final_target_Q": final_target,
        "avg_settled_Q": avg_settled_Q,
        "max_settled_Q": max_settled_Q,
        "min_settled_Q": min_settled_Q,
        "max_achievable_Q": max_achievable,
        "target_achieved": target_achieved,
        "ramp_violations": int(ramp_violations),
        "whp_violations": int(whp_violations),
        "flp_violations": int(flp_violations),
        "bhp_violations": int(bhp_violations),
        "any_safety_violation": any_safety_violation,
    }

    return df, summary


def plot_scenario(df, summary, output_path):
    """Generate a 6-panel plot for a scenario."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

    t = df["Time_hr"]

    # Panel 1: Oil rate (target + actual overlaid)
    ax = axes[0]
    ax.plot(t, df["Target_Q"], "k--", linewidth=1.5, label="Target", alpha=0.8)
    ax.plot(t, df["Actual_Q"], "b-", linewidth=1.0, label="Actual", alpha=0.8)
    ax.set_ylabel("Q (bbl/hr)")
    ax.set_title(f"Scenario {summary['scenario']}: "
                 f"Target={summary['final_target_Q']:.0f} bbl/hr, "
                 f"Achieved={summary['avg_settled_Q']:.1f} bbl/hr")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: WHP
    axes[1].plot(t, df["WHP_psi"], color="tab:orange", linewidth=0.8)
    axes[1].axhline(WHP_MIN, color="red", linestyle=":", alpha=0.5, label="Limits")
    axes[1].axhline(WHP_MAX, color="red", linestyle=":", alpha=0.5)
    axes[1].set_ylabel("WHP (psi)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: FLP
    axes[2].plot(t, df["FLP_psi"], color="tab:green", linewidth=0.8)
    axes[2].axhline(FLP_MIN, color="red", linestyle=":", alpha=0.5, label="Limits")
    axes[2].axhline(FLP_MAX, color="red", linestyle=":", alpha=0.5)
    axes[2].set_ylabel("FLP (psi)")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    # Panel 4: BHP
    axes[3].plot(t, df["BHP_psi"], color="tab:red", linewidth=0.8)
    axes[3].axhline(BHP_MIN, color="red", linestyle=":", alpha=0.5, label="Limits")
    axes[3].axhline(BHP_MAX, color="red", linestyle=":", alpha=0.5)
    axes[3].set_ylabel("BHP (psi)")
    axes[3].legend(loc="upper right", fontsize=8)
    axes[3].grid(True, alpha=0.3)

    # Panel 5: Choke position
    axes[4].step(t, df["Choke_pct"], where="post", color="tab:brown", linewidth=1.2)
    axes[4].set_ylabel("Choke (%)")
    axes[4].set_xlabel("Time (hours)")
    axes[4].set_ylim(-2, 105)
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved plot: {output_path}")


def print_summary(summary):
    """Print a formatted pass/fail summary."""
    s = summary
    print(f"\n  --- Scenario {s['scenario']} Summary ---")

    # Target achievement
    if s["scenario"] == "C":
        # For infeasible target, "achievement" means settling near max achievable
        if s["target_achieved"]:
            print(f"  [PASS] Correctly rejected infeasible target (250 bbl/hr): "
                  f"settled at avg Q = {s['avg_settled_Q']:.1f} bbl/hr "
                  f"(near max ~{s['max_achievable_Q']:.1f})")
        else:
            print(f"  [FAIL] Did not settle near max achievable: "
                  f"avg Q = {s['avg_settled_Q']:.1f} bbl/hr")
    else:
        if s["target_achieved"]:
            print(f"  [PASS] Target achieved: avg Q = {s['avg_settled_Q']:.1f} "
                  f"bbl/hr (target = {s['final_target_Q']:.0f})")
        else:
            print(f"  [FAIL] Target NOT achieved: avg Q = {s['avg_settled_Q']:.1f} "
                  f"bbl/hr (target = {s['final_target_Q']:.0f})")

    # Safety violations
    if not s["any_safety_violation"]:
        print(f"  [PASS] No safety constraint violations (WHP/FLP/BHP all within limits)")
    else:
        print(f"  [FAIL] Safety violations: WHP={s['whp_violations']}, "
              f"FLP={s['flp_violations']}, BHP={s['bhp_violations']}")

    # Ramp rate
    if s["ramp_violations"] == 0:
        print(f"  [PASS] Ramp rate never exceeded +/-{CHOKE_RAMP_MAX}%/step")
    else:
        print(f"  [FAIL] Ramp rate exceeded {s['ramp_violations']} times")


if __name__ == "__main__":
    print("=" * 60)
    print("CLOSED-LOOP SCENARIO RUNNER")
    print("=" * 60)

    # Load identified model and create controller
    model = WellModel.from_step_test(data_path("step_test_full_data.csv"))
    controller = MPCController(model)

    # Create TRUE simulator (with a fixed seed for reproducibility)
    sim = OilWellSimulator(seed=42)

    # ── Scenario A: Startup to Target ──────────────────────────────
    print("\n--- Scenario A: Startup to Target ---")
    print("Starting at idle (choke=10%), target = 130 bbl/hr")
    target_A = [(0, 130.0)]
    df_A, sum_A = run_scenario("A", sim, controller, target_A,
                               start_choke=10.0, n_steps=50)
    df_A.to_csv(data_path("scenario_A_results.csv"), index=False)
    plot_scenario(df_A, sum_A, image_path("scenario_A_results.png"))
    print_summary(sum_A)

    # ── Scenario B: Target Tracking ────────────────────────────────
    print("\n--- Scenario B: Target Tracking ---")
    print("Phase 1: target=100 bbl/hr for 25 hrs, Phase 2: target=150 bbl/hr")
    target_B = [(0, 100.0), (25, 150.0)]
    # Fresh controller for clean logs
    controller_B = MPCController(model)
    df_B, sum_B = run_scenario("B", sim, controller_B, target_B,
                               start_choke=30.0, n_steps=50)
    df_B.to_csv(data_path("scenario_B_results.csv"), index=False)
    plot_scenario(df_B, sum_B, image_path("scenario_B_results.png"))
    print_summary(sum_B)

    # ── Scenario C: Infeasible Target ──────────────────────────────
    print("\n--- Scenario C: Infeasible Target ---")
    print("Target = 250 bbl/hr (well above max achievable ~182 bbl/hr)")
    target_C = [(0, 250.0)]
    controller_C = MPCController(model)
    df_C, sum_C = run_scenario("C", sim, controller_C, target_C,
                               start_choke=30.0, n_steps=40)
    df_C.to_csv(data_path("scenario_C_results.csv"), index=False)
    plot_scenario(df_C, sum_C, image_path("scenario_C_results.png"))
    print_summary(sum_C)

    print("\n" + "=" * 60)
    print("ALL SCENARIOS COMPLETE")
    print("=" * 60)
