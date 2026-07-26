"""
Full-Range Open-Loop Step Test
==============================
Runs a staircase schedule across the full 0-100% choke range using the
simulator, collecting transient and steady-state data for system
identification.

The schedule includes up-steps, down-steps, and reversals so the
identified model captures hysteresis-free behavior across the full
operating range.  Each step is held for 10 hours — enough for the
first-order dynamics (tau ~ 5h) to clearly settle (2× tau reaches
~86% of steady state; 3× tau reaches ~95%).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless saves
import matplotlib.pyplot as plt
from simulator import OilWellSimulator
from paths import data_path, image_path

# ── Staircase schedule ─────────────────────────────────────────────
# (choke_level, hold_hours)
# Includes monotonic ramp-up, down-steps, and reversals so the
# identification dataset exercises the full range from both directions.
SCHEDULE = [
    (0,  20),   # shut-in anchor (4×tau = 20h ensures >98% settled)
    (20, 20),
    (30, 20),   # critical: must match simulator anchor at 30%
    (40, 20),
    (45, 20),
    (55, 20),
    (65, 20),   # top of reference dataset range
    (80, 20),
    (100, 20),  # max choke
    (70, 20),   # down-step
    (50, 20),
    (25, 20),
    (45, 20),   # reversal up (repeat for consistency check)
    (75, 20),
    (35, 20),   # reversal down
    (55, 20),   # reversal up (repeat)
    (10, 20),   # deep down-step
    (90, 20),   # large up-step
    (60, 20),   # moderate down-step
]

SEED = 42
OUTPUT_CSV = data_path("step_test_full_data.csv")
OUTPUT_PNG = image_path("step_test_full_plot.png")


def run_step_test(seed=SEED):
    """Execute the staircase schedule and return a DataFrame."""
    sim = OilWellSimulator(seed=seed)
    # Start from shut-in so the first step captures the full startup transient
    sim.reset(choke_position=0.0)

    records = []
    time = 0.0

    for choke_level, hold_hours in SCHEDULE:
        for _ in range(hold_hours):
            Q, WHP, FLP, BHP = sim.step(choke_level)
            records.append({
                "Time_hr": time,
                "Choke_pct": choke_level,
                "OilRate_bbl_hr": Q,
                "WHP_psi": WHP,
                "FLP_psi": FLP,
                "BHP_psi": BHP,
            })
            time += 1.0

    df = pd.DataFrame(records)
    return df


def sanity_check(df):
    """Validate the step-test data — flag anything suspicious."""
    errors = []

    # No NaNs
    if df.isnull().any().any():
        errors.append("FAIL: NaN values found in dataset")

    # No negative oil rates
    if (df["OilRate_bbl_hr"] < 0).any():
        errors.append("FAIL: Negative oil rates present")

    # No non-positive pressures
    for col in ["WHP_psi", "FLP_psi", "BHP_psi"]:
        if (df[col] <= 0).any():
            errors.append(f"FAIL: Non-positive values in {col}")

    # Check monotonic direction of response at key steps
    # At choke=0, Q should be ~0 (shut-in)
    q_at_zero = df.loc[df["Choke_pct"] == 0, "OilRate_bbl_hr"].mean()
    if q_at_zero > 2.0:
        errors.append(f"WARNING: Q at choke=0 is {q_at_zero:.2f} (expected ~0)")

    # At choke=100, Q should be the maximum observed
    q_at_100 = df.loc[df["Choke_pct"] == 100, "OilRate_bbl_hr"].mean()
    q_at_80 = df.loc[df["Choke_pct"] == 80, "OilRate_bbl_hr"].mean()
    if q_at_100 < q_at_80:
        errors.append(f"FAIL: Q at 100% ({q_at_100:.1f}) < Q at 80% ({q_at_80:.1f})")

    # BHP should decrease with increasing choke (drawdown effect)
    # Check a few levels
    for c1, c2 in [(30, 60), (60, 90)]:
        bhp_c1 = df.loc[df["Choke_pct"] == c1, "BHP_psi"].mean()
        bhp_c2 = df.loc[df["Choke_pct"] == c2, "BHP_psi"].mean()
        # Allow some tolerance for transients
        if bhp_c2 > bhp_c1 + 50:
            errors.append(f"WARNING: BHP at {c2}% ({bhp_c2:.0f}) > BHP at {c1}% ({bhp_c1:.0f})")

    # Overall stats
    total_hours = df["Time_hr"].max() + 1
    n_levels = df["Choke_pct"].nunique()
    print(f"Dataset: {len(df)} rows, {total_hours:.0f} hours, {n_levels} distinct choke levels")
    print(f"  Q range:   [{df['OilRate_bbl_hr'].min():.1f}, {df['OilRate_bbl_hr'].max():.1f}] bbl/hr")
    print(f"  WHP range: [{df['WHP_psi'].min():.1f}, {df['WHP_psi'].max():.1f}] psi")
    print(f"  FLP range: [{df['FLP_psi'].min():.1f}, {df['FLP_psi'].max():.1f}] psi")
    print(f"  BHP range: [{df['BHP_psi'].min():.1f}, {df['BHP_psi'].max():.1f}] psi")

    if errors:
        for e in errors:
            print(f"  {e}")
    else:
        print("  All sanity checks PASSED")
    return errors


def extract_steady_states(df, tail_frac=0.3):
    """
    From each constant-choke segment, extract the last `tail_frac`
    of points as the "steady-state" values (by then the transient
    has mostly settled).
    """
    ss_records = []
    for choke_level, hold_hours in SCHEDULE:
        segment = df[df["Choke_pct"] == choke_level]
        # Take the last portion of the segment
        n_tail = max(3, int(len(segment) * tail_frac))
        tail = segment.tail(n_tail)
        ss_records.append({
            "Choke_pct": choke_level,
            "OilRate_bbl_hr": tail["OilRate_bbl_hr"].mean(),
            "WHP_psi": tail["WHP_psi"].mean(),
            "FLP_psi": tail["FLP_psi"].mean(),
            "BHP_psi": tail["BHP_psi"].mean(),
        })
    return pd.DataFrame(ss_records)


def plot_step_test(df, ss_df):
    """Generate a 5-panel plot: choke, Q, WHP, FLP, BHP vs time."""
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

    t = df["Time_hr"]

    # Choke position
    axes[0].step(t, df["Choke_pct"], where="post", color="tab:brown", linewidth=1.5)
    axes[0].set_ylabel("Choke (%)")
    axes[0].set_title("Full-Range Step Test (0–100% choke)")
    axes[0].grid(True, alpha=0.3)

    # Oil rate
    axes[1].plot(t, df["OilRate_bbl_hr"], color="tab:blue", linewidth=0.8, alpha=0.7, label="Measured")
    axes[1].scatter(ss_df["Choke_pct"] * 0 + ss_df.index * 10 + 5, ss_df["OilRate_bbl_hr"],
                     color="red", zorder=5, s=30, label="Steady-state avg")
    axes[1].set_ylabel("Q (bbl/hr)")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # WHP
    axes[2].plot(t, df["WHP_psi"], color="tab:orange", linewidth=0.8, alpha=0.7)
    axes[2].set_ylabel("WHP (psi)")
    axes[2].grid(True, alpha=0.3)

    # FLP
    axes[3].plot(t, df["FLP_psi"], color="tab:green", linewidth=0.8, alpha=0.7)
    axes[3].set_ylabel("FLP (psi)")
    axes[3].grid(True, alpha=0.3)

    # BHP
    axes[4].plot(t, df["BHP_psi"], color="tab:red", linewidth=0.8, alpha=0.7)
    axes[4].set_ylabel("BHP (psi)")
    axes[4].set_xlabel("Time (hours)")
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150)
    plt.close()
    print(f"Saved plot to {OUTPUT_PNG}")


if __name__ == "__main__":
    print("=" * 60)
    print("FULL-RANGE STEP TEST")
    print("=" * 60)

    df = run_step_test()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved data to {OUTPUT_CSV}")

    errors = sanity_check(df)

    ss_df = extract_steady_states(df)
    print("\nSteady-state values by choke level:")
    print(ss_df.to_string(index=False))

    plot_step_test(df, ss_df)

    if any("FAIL" in e for e in errors):
        print("\n*** STEP TEST FAILED SANITY CHECK — review before proceeding ***")
    else:
        print("\nStep test completed successfully.")
