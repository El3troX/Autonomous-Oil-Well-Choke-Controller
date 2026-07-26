"""
Naturally Flowing Oil Well Simulator
=====================================
Built to match the interface described in the hackathon problem statement:

    Q, WHP, FLP, BHP = simulator.step(choke_position)

CALIBRATION NOTE:
No live simulator was provided as promised in the problem statement. This
simulator was instead calibrated directly against the reference dataset
(Autonomous_Choke_Control_Simulated_Dataset.csv) that WAS provided:

  - Steady-state relationships between choke opening and each of
    Q, WHP, FLP, BHP were extracted from the 5 distinct choke levels
    present in the reference data (30, 40, 45, 55, 65%) and connected
    with monotonic (PCHIP) interpolation, which avoids the overshoot/
    oscillation artifacts of a naive polynomial fit.
  - A shut-in anchor point at choke=0% was added using the intercept
    implied by extrapolating the reference data's linear trend back to
    0% (independently confirmed by both a linear fit and a saturating
    exponential fit, which agreed closely).
  - Two additional anchor points above the tested range (choke=80, 100%)
    were added using a tapering/saturating assumption, since the
    reference data never tested that region. This is an explicit
    modeling assumption, documented here and in the report, needed so
    the controller has sensible (if approximate) behavior when it
    evaluates high-choke candidates.
  - The transient time constant (tau ~ 5 hours) was fit from the
    reference data's step-response transients (two independent step
    segments gave tau = 5.24h and tau = 4.46h; we use 5.0h).

This means the simulator's behavior in the 30-65% choke band -- the
region that was actually tested -- reproduces the reference dataset's
steady-state and transient behavior closely. Outside that band, behavior
is a documented, physically-reasonable extrapolation.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator


class OilWellSimulator:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

        # ---- Anchor points for steady-state curves ----
        # choke=0    : shut-in (extrapolated from reference data trend)
        # choke=30..65: taken directly from reference dataset steady states
        # choke=80,100: tapering assumption beyond tested range (documented)
        self._choke_anchor = np.array([0, 30, 40, 45, 55, 65, 80, 100], dtype=float)

        self._Q_anchor   = np.array([0.0, 93.72, 111.16, 121.00, 139.90, 155.87, 172.0, 182.0])
        self._WHP_anchor = np.array([317.5, 269.92, 259.14, 246.16, 232.37, 217.18, 196.0, 172.0])
        self._FLP_anchor = np.array([216.9, 187.89, 179.63, 174.92, 164.55, 155.28, 140.0, 120.0])
        self._BHP_anchor = np.array([3362.6, 3137.48, 3084.25, 3023.11, 2969.03, 2882.73, 2740.0, 2560.0])

        self._Q_curve   = PchipInterpolator(self._choke_anchor, self._Q_anchor)
        self._WHP_curve = PchipInterpolator(self._choke_anchor, self._WHP_anchor)
        self._FLP_curve = PchipInterpolator(self._choke_anchor, self._FLP_anchor)
        self._BHP_curve = PchipInterpolator(self._choke_anchor, self._BHP_anchor)

        # ---- Dynamics: fit from reference data step-response transients ----
        self.tau_hours = 5.0          # time constant in hours (control interval = 1 hour)
        self.dt_hours = 1.0
        self.alpha = 1 - np.exp(-self.dt_hours / self.tau_hours)  # discrete relaxation factor

        # ---- Measurement noise (std dev), matched to reference data scatter ----
        self.noise_Q = 0.8
        self.noise_WHP = 1.2
        self.noise_FLP = 0.8
        self.noise_BHP = 3.0

        # ---- State ----
        self.choke = 30.0
        Q0, WHP0, FLP0, BHP0 = self._steady_state(self.choke)
        self.Q, self.WHP, self.FLP, self.BHP = Q0, WHP0, FLP0, BHP0

    # ------------------------------------------------------------------
    def _steady_state(self, choke_pct):
        c = np.clip(choke_pct, 0.0, 100.0)
        Q = float(self._Q_curve(c))
        WHP = float(self._WHP_curve(c))
        FLP = float(self._FLP_curve(c))
        BHP = float(self._BHP_curve(c))
        return Q, WHP, FLP, BHP

    # ------------------------------------------------------------------
    def step(self, choke_position):
        """
        Advance the simulator by one control interval (1 hour) given a
        new choke position (0-100%). Returns (Q, WHP, FLP, BHP) after
        first-order relaxation toward the new steady state (tau=5h,
        calibrated from reference data), plus measurement noise matched
        to the reference data's scatter.
        """
        choke_position = float(np.clip(choke_position, 0.0, 100.0))
        self.choke = choke_position

        Q_ss, WHP_ss, FLP_ss, BHP_ss = self._steady_state(choke_position)

        # First-order relaxation toward steady state
        self.Q = self.Q + self.alpha * (Q_ss - self.Q)
        self.WHP = self.WHP + self.alpha * (WHP_ss - self.WHP)
        self.FLP = self.FLP + self.alpha * (FLP_ss - self.FLP)
        self.BHP = self.BHP + self.alpha * (BHP_ss - self.BHP)

        # Measurement noise
        Q_meas = max(self.Q + self.rng.normal(0, self.noise_Q), 0.0)
        WHP_meas = self.WHP + self.rng.normal(0, self.noise_WHP)
        FLP_meas = self.FLP + self.rng.normal(0, self.noise_FLP)
        BHP_meas = self.BHP + self.rng.normal(0, self.noise_BHP)

        return Q_meas, WHP_meas, FLP_meas, BHP_meas

    def reset(self, choke_position=30.0):
        self.choke = choke_position
        Q0, WHP0, FLP0, BHP0 = self._steady_state(choke_position)
        self.Q, self.WHP, self.FLP, self.BHP = Q0, WHP0, FLP0, BHP0
        return self.Q, self.WHP, self.FLP, self.BHP


# Convenience module-level instance, matching `simulator.step(...)` usage
simulator = OilWellSimulator()


if __name__ == "__main__":
    # Quick sanity check: step through a few choke positions
    sim = OilWellSimulator()
    for choke in [10, 10, 10, 30, 30, 30, 30, 60, 60, 60, 60, 90, 90, 90, 90]:
        Q, WHP, FLP, BHP = sim.step(choke)
        print(f"choke={choke:5.1f}%  Q={Q:7.2f} bbl/hr  WHP={WHP:7.1f} psi  "
              f"FLP={FLP:7.1f} psi  BHP={BHP:7.1f} psi")
