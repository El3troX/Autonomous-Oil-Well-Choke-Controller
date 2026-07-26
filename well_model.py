"""
Well Predictive Model for MPC
=============================
A lightweight first-order dynamic model identified from step-test data,
used by the MPC controller to forecast system behavior over a planning
horizon.

This is NOT the same object as the true simulator — it's a black-box
model identified from input/output data, which is the correct practice
for model-based control.

Model structure:
  - Steady-state: PCHIP interpolation over observed choke levels
  - Dynamics:     First-order relaxation  x(k+1) = x(k) + α*(x_ss - x(k))
                  where α = 1 - exp(-dt/τ), τ identified from step transients
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit
from paths import data_path, resolve_data_path


class WellModel:
    """
    First-order dynamic model identified from step-test data.

    Usage:
        model = WellModel.from_step_test(data_path("step_test_full_data.csv"))
        preds = model.predict(current_state, choke_sequence, horizon)
    """

    def __init__(self):
        # Steady-state PCHIP curves (fitted later)
        self._Q_curve = None
        self._WHP_curve = None
        self._FLP_curve = None
        self._BHP_curve = None

        # Identified dynamics
        self.tau = None       # time constant (hours)
        self.dt = 1.0         # control interval (hours)
        self.alpha = None     # discrete relaxation factor

        # Steady-state lookup table (for reference)
        self.ss_table = None
        self._steady_state_fitted = False
        self._dynamics_fitted = False

    def _require_steady_state(self):
        if not self._steady_state_fitted:
            raise RuntimeError("steady-state model has not been fitted yet")

    def _require_dynamics(self):
        if not self._dynamics_fitted:
            raise RuntimeError("dynamic model has not been fitted yet")

    def _validate_state(self, current_state):
        required_keys = ("Q", "WHP", "FLP", "BHP", "choke")
        missing = [key for key in required_keys if key not in current_state]
        if missing:
            raise ValueError(f"current_state is missing required keys: {missing}")

        validated = {}
        for key in required_keys:
            value = float(current_state[key])
            if not np.isfinite(value):
                raise ValueError(f"current_state['{key}'] must be finite")
            validated[key] = value
        return validated

    # ── Fitting ────────────────────────────────────────────────────
    def fit_steady_state(self, df, tail_frac=0.3):
        """
        Extract steady-state values from each constant-choke segment
        and fit PCHIP monotonic interpolators.
        """
        # Group by constant-choke segments
        df = df.copy()
        df["segment"] = (df["Choke_pct"].diff().abs() > 0.01).cumsum()

        ss_records = []
        for seg_id, seg in df.groupby("segment"):
            choke = seg["Choke_pct"].iloc[0]
            n_tail = max(3, int(len(seg) * tail_frac))
            tail = seg.tail(n_tail)
            ss_records.append({
                "Choke_pct": choke,
                "Q_ss": tail["OilRate_bbl_hr"].mean(),
                "WHP_ss": tail["WHP_psi"].mean(),
                "FLP_ss": tail["FLP_psi"].mean(),
                "BHP_ss": tail["BHP_psi"].mean(),
            })

        ss_df = pd.DataFrame(ss_records)
        # Aggregate duplicates (e.g. choke=60 visited twice) by averaging
        ss_df = ss_df.groupby("Choke_pct", as_index=False).mean()
        ss_df = ss_df.sort_values("Choke_pct").reset_index(drop=True)
        self.ss_table = ss_df

        choke = ss_df["Choke_pct"].values
        self._Q_curve = PchipInterpolator(choke, ss_df["Q_ss"].values)
        self._WHP_curve = PchipInterpolator(choke, ss_df["WHP_ss"].values)
        self._FLP_curve = PchipInterpolator(choke, ss_df["FLP_ss"].values)
        self._BHP_curve = PchipInterpolator(choke, ss_df["BHP_ss"].values)
        self._steady_state_fitted = True

        print(f"Fitted steady-state curves from {len(ss_df)} choke levels: "
              f"{choke.tolist()}")

    def fit_dynamics(self, df):
        """
        Estimate the first-order relaxation time constant from step
        transients in the data.

        Method: For each step change, use the MEASURED steady-state
        (tail average of the *next* segment, once it's settled) as
        the target — NOT the PCHIP curve, which may have small errors.
        This makes the tau estimate self-consistent.

        We fit Q (oil rate) as it's the primary variable with cleanest
        transient signal. Use least-squares over the first 8 time steps
        of each transition (the most informative part of the transient).
        """
        df = df.copy()
        df["segment"] = (df["Choke_pct"].diff().abs() > 0.01).cumsum()

        segments = list(df.groupby("segment"))
        taus = []

        for i in range(1, len(segments)):
            prev_seg = segments[i - 1][1]
            curr_seg = segments[i][1]

            if len(curr_seg) < 5:
                continue

            # Initial state: last measured Q of previous segment (with noise)
            x_init = prev_seg["OilRate_bbl_hr"].iloc[-1]

            # Measured target: average of last 40% of current segment
            # (by then transient has settled, giving a robust target)
            n_tail = max(3, int(len(curr_seg) * 0.4))
            x_final = curr_seg["OilRate_bbl_hr"].tail(n_tail).mean()

            if abs(x_final - x_init) < 5.0:
                continue  # skip tiny steps — signal too weak for reliable fit

            # Transient trajectory
            x_data = curr_seg["OilRate_bbl_hr"].values
            t_data = np.arange(len(x_data), dtype=float)

            # Deviation from final: y(t) = (x_init - x_final) * exp(-t/tau)
            init_dev = x_init - x_final
            deviation = x_data - x_final

            # Fit using only first 8 steps (most informative region)
            n_fit = min(8, len(t_data))
            t_fit = t_data[:n_fit]
            dev_fit = deviation[:n_fit]

            # Least-squares on log-transformed deviation
            # Skip points where deviation sign flips (noise)
            abs_dev = np.abs(dev_fit)
            if abs_dev.max() < 1.0:
                continue

            def exp_decay(t, tau_est):
                return init_dev * np.exp(-t / tau_est)

            try:
                popt, _ = curve_fit(exp_decay, t_fit, dev_fit,
                                    p0=[5.0], bounds=(1.0, 20.0))
                taus.append(popt[0])
            except (RuntimeError, ValueError):
                pass

        if taus:
            self.tau = np.median(taus)  # median is robust to outliers
        else:
            self.tau = 5.0  # fallback
            print("WARNING: Could not fit tau from transients, using default 5.0h")

        self.alpha = 1.0 - np.exp(-self.dt / self.tau)
        self._dynamics_fitted = True
        print(f"Identified dynamics: tau = {self.tau:.2f} hours, "
              f"alpha = {self.alpha:.4f}")
        print(f"  (fitted from {len(taus)} step transitions, "
              f"individual taus: {[f'{t:.2f}' for t in taus]})")

    def fit(self, df, tail_frac=0.3):
        """Fit both steady-state and dynamics from step-test data."""
        self.fit_steady_state(df, tail_frac=tail_frac)
        self.fit_dynamics(df)

    # ── Steady-state evaluation ────────────────────────────────────
    def steady_state(self, choke_pct):
        """Return (Q_ss, WHP_ss, FLP_ss, BHP_ss) for a given choke."""
        self._require_steady_state()
        c = np.clip(float(choke_pct), 0.0, 100.0)
        return (float(self._Q_curve(c)),
                float(self._WHP_curve(c)),
                float(self._FLP_curve(c)),
                float(self._BHP_curve(c)))

    # ── Prediction ─────────────────────────────────────────────────
    def predict(self, current_state, choke_sequence, horizon=None):
        """
        Predict system trajectories over a sequence of future choke moves.

        Parameters
        ----------
        current_state : dict
            Current system state with keys: Q, WHP, FLP, BHP, choke
        choke_sequence : array-like
            Sequence of future choke positions to apply (length = horizon).
        horizon : int, optional
            Prediction horizon. If None, defaults to len(choke_sequence).

        Returns
        -------
        dict with arrays: Q, WHP, FLP, BHP (each of length horizon+1,
        including the current state at index 0).
        """
        self._require_steady_state()
        self._require_dynamics()

        current_state = self._validate_state(current_state)
        choke_sequence = np.asarray(choke_sequence, dtype=float)

        if horizon is None:
            horizon = len(choke_sequence)
        if horizon < 0:
            raise ValueError("horizon must be non-negative")
        if len(choke_sequence) != horizon:
            raise ValueError(
                f"choke_sequence length ({len(choke_sequence)}) must match horizon ({horizon})"
            )

        # Initialize trajectories with current state
        Q = np.zeros(horizon + 1)
        WHP = np.zeros(horizon + 1)
        FLP = np.zeros(horizon + 1)
        BHP = np.zeros(horizon + 1)

        Q[0] = current_state["Q"]
        WHP[0] = current_state["WHP"]
        FLP[0] = current_state["FLP"]
        BHP[0] = current_state["BHP"]

        for k in range(horizon):
            choke_k = float(np.clip(choke_sequence[k], 0.0, 100.0))
            Q_ss, WHP_ss, FLP_ss, BHP_ss = self.steady_state(choke_k)

            Q[k + 1] = Q[k] + self.alpha * (Q_ss - Q[k])
            WHP[k + 1] = WHP[k] + self.alpha * (WHP_ss - WHP[k])
            FLP[k + 1] = FLP[k] + self.alpha * (FLP_ss - FLP[k])
            BHP[k + 1] = BHP[k] + self.alpha * (BHP_ss - BHP[k])

        return {"Q": Q, "WHP": WHP, "FLP": FLP, "BHP": BHP}

    # ── Validation ─────────────────────────────────────────────────
    def validate(self, df, holdout_frac=0.4):
        """
        Validate the identified model against held-out step-test data.
        Uses the last `holdout_frac` of each segment for validation.

        Parameters
        ----------
        df : str or DataFrame
            Path to CSV or DataFrame with step-test data.
        holdout_frac : float
            Fraction of each segment held out for validation.
        """
        self._require_steady_state()
        self._require_dynamics()

        if isinstance(df, (str, Path)):
            df = pd.read_csv(resolve_data_path(df))
        df = df.copy()
        df["segment"] = (df["Choke_pct"].diff().abs() > 0.01).cumsum()

        all_errors = {"Q": [], "WHP": [], "FLP": [], "BHP": []}

        for seg_id, seg in df.groupby("segment"):
            choke = seg["Choke_pct"].iloc[0]
            n_train = int(len(seg) * (1 - holdout_frac))
            train_seg = seg.iloc[:n_train]
            holdout_seg = seg.iloc[n_train:]

            if len(holdout_seg) < 2:
                continue

            # Use last training point as initial state
            state = {
                "Q": train_seg["OilRate_bbl_hr"].iloc[-1],
                "WHP": train_seg["WHP_psi"].iloc[-1],
                "FLP": train_seg["FLP_psi"].iloc[-1],
                "BHP": train_seg["BHP_psi"].iloc[-1],
                "choke": choke,
            }

            # Predict forward using constant choke
            n_pred = len(holdout_seg)
            preds = self.predict(state, [choke] * n_pred)

            # Compare (skip index 0 which is the initial state)
            for var, col in [("Q", "OilRate_bbl_hr"),
                             ("WHP", "WHP_psi"),
                             ("FLP", "FLP_psi"),
                             ("BHP", "BHP_psi")]:
                actual = holdout_seg[col].values
                predicted = preds[var][1:]  # skip initial state
                min_len = min(len(actual), len(predicted))
                all_errors[var].extend((actual[:min_len] - predicted[:min_len]).tolist())

        # Report RMSE per variable
        print("\nModel Validation Report (hold-out data):")
        print("-" * 50)
        rmse = {}
        for var in ["Q", "WHP", "FLP", "BHP"]:
            err = np.array(all_errors[var])
            rmse[var] = np.sqrt(np.mean(err ** 2))
            mae = np.mean(np.abs(err))
            print(f"  {var:4s}: RMSE = {rmse[var]:7.2f}, MAE = {mae:7.2f} "
                  f"({len(err)} comparison points)")
        print("-" * 50)
        return rmse

    # ── Factory ────────────────────────────────────────────────────
    @classmethod
    def from_step_test(cls, csv_path, tail_frac=0.3):
        """Load step-test data and fit the model."""
        df = pd.read_csv(resolve_data_path(csv_path))
        model = cls()
        model.fit(df, tail_frac=tail_frac)
        return model


# ── Quick test when run directly ───────────────────────────────────
if __name__ == "__main__":
    model = WellModel.from_step_test(data_path("step_test_full_data.csv"))

    print("\nSteady-state lookup:")
    for c in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        q, whp, flp, bhp = model.steady_state(c)
        print(f"  choke={c:3d}%  Q={q:7.1f}  WHP={whp:6.1f}  "
              f"FLP={flp:6.1f}  BHP={bhp:6.0f}")

    # Validate
    model.validate(data_path("step_test_full_data.csv"))

    # Quick predict test
    state = {"Q": 90.0, "WHP": 270.0, "FLP": 188.0, "BHP": 3137.0, "choke": 30.0}
    preds = model.predict(state, [50.0] * 5)
    print("\n5-step prediction from choke=30->50:")
    for k in range(6):
        print(f"  step {k}: Q={preds['Q'][k]:.1f}, WHP={preds['WHP'][k]:.1f}, "
              f"FLP={preds['FLP'][k]:.1f}, BHP={preds['BHP'][k]:.0f}")
