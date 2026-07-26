"""
Shared Configuration — Safety Envelope & Controller Parameters
=============================================================
Safe operating limits for WHP, FLP, BHP based on the full-range
step test data plus conservative margins.

REASONING FOR MARGINS:

Observed ranges from step_test_full.py (full 0-100% choke sweep):
  WHP:  [171, 319] psi
  FLP:  [120, 219] psi
  BHP:  [2560, 3366] psi

Margins are applied INWARD from the observed extremes, not at the
extremes themselves.  This ensures:
  1. The MPC rejects candidates that would push into untested territory.
  2. Transient overshoots (which are inherent in first-order dynamics
     after a large step) don't immediately violate constraints.
  3. There is headroom for the controller to maneuver before hitting
     hard limits.

WHP  — min: 180 psi (well below minimum observed at choke=100%,
       which is ~171 psi; 9 psi margin for transient undershoot)
       — max: 310 psi (well below max at choke=0%, ~319 psi;
       9 psi margin for transient overshoot during startup)

FLP  — min: 130 psi (observed min ~120 psi at choke=100%;
       10 psi margin — FLP tracks WHP but at lower absolute level,
       so a 10 psi floor is conservative)
       — max: 210 psi (observed max ~219 psi at choke=0%;
       9 psi margin)

BHP  — min: 2600 psi (observed min ~2560 psi at choke=100%;
       40 psi margin — BHP has larger absolute values so the
       margin is proportionally similar at ~1.5%)
       — max: 3350 psi (observed max ~3366 psi at choke=0%;
       16 psi margin, ~0.5%)

These are deliberately asymmetric in absolute terms because the
underlying physics are nonlinear — the relationships are smooth but
not symmetric around any midpoint.
"""

# ── Safety Envelope (psi) ──────────────────────────────────────────
WHP_MIN = 180.0     # psi — below this, risk of well instability
WHP_MAX = 310.0     # psi — above this, risk of equipment overpressure

FLP_MIN = 130.0     # psi — below this, flowline may slug/instability
FLP_MAX = 210.0     # psi — above this, flowline overpressure risk

BHP_MIN = 2600.0    # psi — below this, sand production / drawdown risk
BHP_MAX = 3350.0    # psi — above this, reservoir damage risk

# ── Controller Parameters ──────────────────────────────────────────
CHOKE_MIN = 0.0             # % — minimum choke position
CHOKE_MAX = 100.0           # % — maximum choke position
CHOKE_RAMP_MAX = 5.0        # % — maximum single-step choke movement
CHOKE_RESOLUTION = 1.0      # % — MPC candidate evaluation resolution
MPC_HORIZON = 4             # steps — prediction horizon for MPC

# ── Noise Characterization (from sample_dataset.csv.csv) ───────────
# Used by the synthetic dataset generator.  These represent the
# standard deviation of measurement noise around the smooth trend.
NOISE_Q_STD = 0.8           # bbl/hr
NOISE_WHP_STD = 1.2         # psi
NOISE_FLP_STD = 0.8         # psi
NOISE_BHP_STD = 3.0         # psi


def check_safety(whp, flp, bhp):
    """
    Check whether a set of pressure readings is within safe limits.

    Returns
    -------
    safe : bool
        True if all pressures are within the envelope.
    violations : list of str
        Descriptions of any violated constraints.
    """
    violations = []
    if whp < WHP_MIN:
        violations.append(f"WHP={whp:.1f} < {WHP_MIN} (min)")
    if whp > WHP_MAX:
        violations.append(f"WHP={whp:.1f} > {WHP_MAX} (max)")
    if flp < FLP_MIN:
        violations.append(f"FLP={flp:.1f} < {FLP_MIN} (min)")
    if flp > FLP_MAX:
        violations.append(f"FLP={flp:.1f} > {FLP_MAX} (max)")
    if bhp < BHP_MIN:
        violations.append(f"BHP={bhp:.1f} < {BHP_MIN} (min)")
    if bhp > BHP_MAX:
        violations.append(f"BHP={bhp:.1f} > {BHP_MAX} (max)")
    return len(violations) == 0, violations


if __name__ == "__main__":
    print("Safety Envelope:")
    print(f"  WHP: [{WHP_MIN}, {WHP_MAX}] psi")
    print(f"  FLP: [{FLP_MIN}, {FLP_MAX}] psi")
    print(f"  BHP: [{BHP_MIN}, {BHP_MAX}] psi")
    print(f"  Choke: [{CHOKE_MIN}, {CHOKE_MAX}]%, ramp <= {CHOKE_RAMP_MAX}%/step")
    print(f"  MPC horizon: {MPC_HORIZON} steps")

    # Quick sanity check
    ok, v = check_safety(250, 170, 3000)
    print(f"\n  Check (250, 170, 3000): safe={ok}, violations={v}")
    ok, v = check_safety(170, 220, 3400)
    print(f"  Check (170, 220, 3400): safe={ok}, violations={v}")
