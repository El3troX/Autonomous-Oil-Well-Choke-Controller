"""
Brute-Force MPC Choke Controller
=================================
At each control step:
  1. Generate candidate choke moves within +/- CHOKE_RAMP_MAX of the
     current position (clamped to [0, 100]).
  2. For each candidate, simulate forward MPC_HORIZON steps using the
     identified WellModel.
  3. Reject any candidate whose predicted trajectory violates the
     safety envelope (WHP, FLP, BHP).
  4. Among feasible candidates, pick the one whose predicted Q at the
     end of the horizon is closest to the target rate.
  5. If NO candidate is feasible (edge case), hold position and log
     a warning.
"""

import logging
import numpy as np
from well_model import WellModel
from config import (
    CHOKE_MIN, CHOKE_MAX, CHOKE_RAMP_MAX, CHOKE_RESOLUTION,
    MPC_HORIZON, WHP_MIN, WHP_MAX, FLP_MIN, FLP_MAX, BHP_MIN, BHP_MAX,
)
from paths import data_path

logger = logging.getLogger(__name__)


class MPCController:
    """
    Model Predictive Controller for oil well choke management.

    Parameters
    ----------
    model : WellModel
        The identified predictive model (NOT the true simulator).
    horizon : int
        Prediction horizon in control steps.
    ramp_max : float
        Maximum choke change per step (%).
    resolution : float
        Resolution for candidate choke evaluation (%).
    safety_margin : dict or None
        Margins subtracted from safety limits when evaluating predictions
        (accounts for model identification error and transient undershoot).
        Keys: WHP_min_margin, WHP_max_margin, FLP_min_margin, etc.
        If None, uses defaults (5.0 psi for WHP/FLP, 20 psi for BHP).
    """

    def __init__(self, model, horizon=MPC_HORIZON,
                 ramp_max=CHOKE_RAMP_MAX, resolution=CHOKE_RESOLUTION,
                 safety_margin=None):
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if ramp_max <= 0:
            raise ValueError("ramp_max must be positive")
        if resolution <= 0:
            raise ValueError("resolution must be positive")

        self.model = model
        self.horizon = horizon
        self.ramp_max = ramp_max
        self.resolution = resolution
        # Safety margins to account for model mismatch and transient effects
        default_margin = {
            "WHP_min_margin": 5.0,
            "WHP_max_margin": 5.0,
            "FLP_min_margin": 5.0,
            "FLP_max_margin": 5.0,
            "BHP_min_margin": 20.0,
            "BHP_max_margin": 20.0,
        }
        self.safety_margin = safety_margin if safety_margin else default_margin

        # Pre-compute the candidate choke offsets
        self._offsets = np.arange(-ramp_max, ramp_max + resolution * 0.5,
                                  resolution)

        # Logging
        self._log_entries = []

    def _validate_current_state(self, current_state):
        """Validate the required controller inputs before optimization."""
        required_keys = ("Q", "WHP", "FLP", "BHP", "choke")
        missing = [key for key in required_keys if key not in current_state]
        if missing:
            raise ValueError(f"current_state is missing required keys: {missing}")

        for key in required_keys:
            value = float(current_state[key])
            if not np.isfinite(value):
                raise ValueError(f"current_state['{key}'] must be finite")

        return {key: float(current_state[key]) for key in required_keys}

    def _generate_candidates(self, current_choke):
        """Generate all feasible choke moves from current position."""
        candidates = current_choke + self._offsets
        candidates = np.clip(candidates, CHOKE_MIN, CHOKE_MAX)
        # Remove duplicates (can happen at boundaries)
        candidates = np.unique(candidates)
        return candidates

    def _trajectory_limits(self, use_margins=True):
        """Return lower/upper safety bounds, optionally with back-off margins."""
        if use_margins:
            m = self.safety_margin
            return {
                "WHP": (WHP_MIN + m["WHP_min_margin"], WHP_MAX - m["WHP_max_margin"]),
                "FLP": (FLP_MIN + m["FLP_min_margin"], FLP_MAX - m["FLP_max_margin"]),
                "BHP": (BHP_MIN + m["BHP_min_margin"], BHP_MAX - m["BHP_max_margin"]),
            }
        return {
            "WHP": (WHP_MIN, WHP_MAX),
            "FLP": (FLP_MIN, FLP_MAX),
            "BHP": (BHP_MIN, BHP_MAX),
        }

    def _check_trajectory_safety(self, predicted, use_margins=True):
        """
        Check whether an entire predicted trajectory is safe.

        Uses safety margins to create a conservative back-off from the
        hard limits, accounting for model identification error and
        transient undershoot/overshoot effects.

        Parameters
        ----------
        predicted : dict
            Arrays of Q, WHP, FLP, BHP over the horizon (length horizon+1).

        Returns
        -------
        safe : bool
        """
        limits = self._trajectory_limits(use_margins=use_margins)

        for k in range(1, len(predicted["WHP"])):
            whp_lo, whp_hi = limits["WHP"]
            flp_lo, flp_hi = limits["FLP"]
            bhp_lo, bhp_hi = limits["BHP"]

            if predicted["WHP"][k] < whp_lo or predicted["WHP"][k] > whp_hi:
                return False
            if predicted["FLP"][k] < flp_lo or predicted["FLP"][k] > flp_hi:
                return False
            if predicted["BHP"][k] < bhp_lo or predicted["BHP"][k] > bhp_hi:
                return False
        return True

    def _trajectory_violation_score(self, predicted, use_margins=True):
        """Measure how far a trajectory is outside the chosen safety envelope."""
        limits = self._trajectory_limits(use_margins=use_margins)
        violation = 0.0

        for k in range(1, len(predicted["WHP"])):
            for name in ("WHP", "FLP", "BHP"):
                lo, hi = limits[name]
                value = float(predicted[name][k])
                if value < lo:
                    violation += lo - value
                elif value > hi:
                    violation += value - hi

        return violation

    def compute_action(self, current_state, target_Q):
        """
        Compute the next choke position given the current state and
        target oil production rate.

        Parameters
        ----------
        current_state : dict
            Keys: Q, WHP, FLP, BHP, choke (all floats).
        target_Q : float
            Desired oil production rate (bbl/hr).

        Returns
        -------
        next_choke : float
            The recommended choke position (%).
        info : dict
            Diagnostic information (feasible count, best error, etc.).
        """
        current_state = self._validate_current_state(current_state)
        target_Q = float(target_Q)
        if not np.isfinite(target_Q):
            raise ValueError("target_Q must be finite")

        current_choke = current_state["choke"]
        candidates = self._generate_candidates(current_choke)

        best_choke = current_choke  # default: hold position
        best_error = float("inf")
        best_move = float("inf")
        best_margin_violation = 0.0
        feasible_count = 0
        all_candidates_infeasible = True
        used_hard_limit_fallback = False

        for cand in candidates:
            # Build choke sequence: hold this candidate for all horizon steps
            choke_seq = [cand] * self.horizon

            # Predict trajectory
            pred = self.model.predict(current_state, choke_seq, self.horizon)

            # Check safety
            if not self._check_trajectory_safety(pred, use_margins=True):
                continue

            all_candidates_infeasible = False
            feasible_count += 1

            # Score: how close is the final predicted Q to target?
            final_Q = pred["Q"][-1]
            error = abs(final_Q - target_Q)
            move = abs(cand - current_choke)
            score = (error, move)

            # Prefer smaller choke movement when prediction error ties.
            if score < (best_error, best_move):
                best_error = error
                best_move = move
                best_margin_violation = 0.0
                best_choke = cand

        # Edge case: no feasible candidate found
        if all_candidates_infeasible:
            hard_safe_candidates = []
            for cand in candidates:
                choke_seq = [cand] * self.horizon
                pred = self.model.predict(current_state, choke_seq, self.horizon)
                if self._check_trajectory_safety(pred, use_margins=False):
                    hard_safe_candidates.append((cand, pred))

            if hard_safe_candidates:
                used_hard_limit_fallback = True
                fallback_score = (float("inf"), float("inf"), float("inf"))
                for cand, pred in hard_safe_candidates:
                    error = abs(pred["Q"][-1] - target_Q)
                    move = abs(cand - current_choke)
                    margin_violation = self._trajectory_violation_score(pred, use_margins=True)
                    score = (margin_violation, error, move)
                    if score < fallback_score:
                        fallback_score = score
                        best_margin_violation = margin_violation
                        best_error = error
                        best_move = move
                        best_choke = cand

                logger.warning(
                    f"No candidate met the conservative back-off at choke={current_choke:.1f}%, "
                    f"target_Q={target_Q:.1f}. Using hard-limit-safe fallback."
                )
            else:
                logger.warning(
                    f"No feasible choke candidate at choke={current_choke:.1f}%, "
                    f"target_Q={target_Q:.1f}. Holding position."
                )
                best_choke = current_choke

        info = {
            "feasible_candidates": feasible_count,
            "total_candidates": len(candidates),
            "best_error": best_error,
            "best_move": best_move,
            "best_margin_violation": best_margin_violation,
            "all_infeasible": all_candidates_infeasible,
            "used_hard_limit_fallback": used_hard_limit_fallback,
        }

        return best_choke, info

    def log_step(self, time, choke, target_Q, actual_Q, whp, flp, bhp, info):
        """Log a control step for post-run analysis."""
        self._log_entries.append({
            "Time_hr": time,
            "Choke_pct": choke,
            "Target_Q": target_Q,
            "Actual_Q": actual_Q,
            "WHP_psi": whp,
            "FLP_psi": flp,
            "BHP_psi": bhp,
            "Feasible_Candidates": info["feasible_candidates"],
            "Best_Error": info["best_error"],
            "Best_Move": info["best_move"],
            "Margin_Violation": info["best_margin_violation"],
            "Fallback_Used": info["used_hard_limit_fallback"],
        })

    def get_log_df(self):
        """Return the logged data as a DataFrame."""
        import pandas as pd
        return pd.DataFrame(self._log_entries)


# ── Quick test when run directly ───────────────────────────────────
if __name__ == "__main__":
    from simulator import OilWellSimulator

    print("MPC Controller Smoke Test")
    print("=" * 50)

    # Load model
    model = WellModel.from_step_test(data_path("step_test_full_data.csv"))

    # Create controller
    ctrl = MPCController(model)

    # Create simulator
    sim = OilWellSimulator(seed=42)

    # Test from startup
    sim.reset(choke_position=10.0)
    Q, WHP, FLP, BHP = sim.step(10.0)

    target = 120.0
    print(f"\nStarting: choke=10%, target_Q={target} bbl/hr")
    print(f"  Initial: Q={Q:.1f}, WHP={WHP:.1f}, FLP={FLP:.1f}, BHP={BHP:.0f}")

    current_choke = 10.0
    for step in range(20):
        state = {"Q": Q, "WHP": WHP, "FLP": FLP, "BHP": BHP, "choke": current_choke}
        next_choke, info = ctrl.compute_action(state, target)
        Q, WHP, FLP, BHP = sim.step(next_choke)
        current_choke = next_choke
        print(f"  step {step+1:2d}: choke={next_choke:5.1f}% -> "
              f"Q={Q:6.1f}  WHP={WHP:6.1f}  FLP={FLP:6.1f}  BHP={BHP:6.0f}  "
              f"(feasible={info['feasible_candidates']}/{info['total_candidates']})")

    print("\nDone.")
