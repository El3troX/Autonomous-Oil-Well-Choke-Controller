# Autonomous Production Choke Controller — Results Summary

## File Index

| File | Purpose |
|------|---------|
| `simulator.py` | Oil well simulator calibrated against reference dataset. PCHIP steady-state curves, first-order dynamics (tau=5h), Gaussian noise. **Do not modify.** |
| `config.py` | Shared safety envelope (WHP/FLP/BHP limits) and controller parameters |
| `step_test_full.py` | Full-range (0-100%) open-loop step test, saves `data/step_test_full_data.csv` |
| `data/step_test_full_data.csv` | Step-test data: 17 choke levels, 20h hold each, with reversals |
| `images/step_test_full_plot.png` | 5-panel plot of the step test (choke/Q/WHP/FLP/BHP vs time) |
| `well_model.py` | First-order dynamic model identified from step-test data. PCHIP steady-state curves + exponential relaxation dynamics |
| `controller.py` | Brute-force MPC controller. Evaluates candidate choke moves over a 4-step horizon, enforces safety constraints with margins |
| `run_scenarios.py` | Closed-loop scenario runner: runs all 3 demonstration scenarios against the true simulator |
| `data/scenario_A_results.csv` / `images/scenario_A_results.png` | Scenario A (Startup to Target) results |
| `data/scenario_B_results.csv` / `images/scenario_B_results.png` | Scenario B (Target Tracking) results |
| `data/scenario_C_results.csv` / `images/scenario_C_results.png` | Scenario C (Infeasible Target) results |
| `generate_synthetic_dataset.py` | Synthetic dataset generator matching sample_dataset.csv style |
| `data/synthetic_choke_dataset.csv` | Generated synthetic dataset (seed=42, 400 hours, full 0-100% range) |
| `images/synthetic_dataset_validation.png` | Validation plot comparing synthetic vs original dataset |
| `tests/test_simulator.py` | Simulator validation tests (no NaN, no negative Q, convergence) |
| `tests/test_well_model.py` | Model prediction and validation tests (shape, RMSE, monotonicity) |
| `tests/test_controller.py` | Controller constraint tests (ramp rate, bounds, target-seeking) |
| `RESULTS_SUMMARY.md` | This file |

---

## Model Validation Results (Phase 2)

The well model was identified from step-test data covering the full 0-100% choke range. Key parameters:

| Parameter | Identified Value | True Value (simulator) | Notes |
|-----------|-----------------|----------------------|-------|
| tau (hours) | 3.52 | 5.0 | Faster response due to noise + incomplete settling in step test |
| alpha (discrete) | 0.2474 | 0.1813 | Derived from identified tau |
| Steady-state method | PCHIP interpolation | PCHIP interpolation | Same approach, different anchor values |

**Hold-out Validation RMSE** (40% of each segment held out for testing):

| Variable | RMSE | MAE | Units | % of Range |
|----------|------|-----|-------|-----------|
| Q (Oil Rate) | 0.95 | 0.76 | bbl/hr | 0.5% of 180 bbl/hr range |
| WHP | 1.35 | 0.99 | psi | 0.6% of 220 psi range |
| FLP | 0.98 | 0.79 | psi | 0.5% of 190 psi range |
| BHP | 4.40 | 3.48 | psi | 0.4% of 1100 psi range |

The model achieves sub-1 bbl/hr accuracy on the primary variable (Q) and less than 1% error on all pressure variables relative to their operating ranges.

---

## Safety Envelope (Phase 3)

Safe operating limits with margins for model uncertainty and transient effects:

| Variable | Minimum | Maximum | Margin Applied | Rationale |
|----------|---------|---------|----------------|-----------|
| WHP (psi) | 180 | 310 | 5 psi each side | Well below observed extremes (171-319), 5 psi back-off for model mismatch |
| FLP (psi) | 130 | 210 | 5 psi each side | Observed range 120-219, margins prevent transient violations |
| BHP (psi) | 2600 | 3350 | 20 psi each side | Proportional to absolute level (~0.6%), accounts for BHP noise (std=3.0) |
| Choke (%) | 0 | 100 | N/A | Hard physical limits |
| Ramp (%/step) | ±5 | — | N/A | Maximum single-step choke movement |

---

## Scenario Results (Phase 5)

All scenarios were run against the **true simulator** (not the identified model), with safety margins applied in the MPC constraint checks.

### Scenario A: Startup to Target

**Objective:** Start at idle (choke=10%) and bring the well to 130 bbl/hr.

- **Starting conditions:** Q ≈ 38 bbl/hr, choke = 10%
- **Target Q:** 130 bbl/hr
- **Settled Q (last 20% of run):** 129.5 bbl/hr (0.4% error)
- **Ramp rate:** Never exceeded ±5%/step
- **Safety violations:** None (WHP, FLP, BHP all within limits)
- **Behavior:** Controller initially used the hard-limit-safe fallback because the startup state sat outside the conservative back-off band, then ramped choke from 10% → 45% over ~18 steps and fine-tuned around 45-47% to maintain target. Smooth, well-behaved approach with no overshoot.

**PASS — all criteria met.**

### Scenario B: Target Tracking

**Objective:** Track a production target that changes mid-run (100 → 150 bbl/hr).

- **Phase 1 (0-25h):** Target = 100 bbl/hr, started from choke=30%
- **Phase 2 (25-50h):** Target switched to 150 bbl/hr
- **Settled Q Phase 1:** ~100 bbl/hr (correctly tracked)
- **Settled Q Phase 2:** 149.2 bbl/hr (0.5% error from 150 target)
- **Ramp rate:** Never exceeded ±5%/step
- **Safety violations:** None
- **Behavior:** Smooth transition when target changed. Controller smoothly increased choke to meet the new higher target, respecting all constraints throughout.

**PASS — all criteria met.**

### Scenario C: Infeasible Target

**Objective:** Request a target (250 bbl/hr) that exceeds the maximum achievable rate (~182 bbl/hr). Controller must reject the target and settle at the maximum safe rate.

- **Target Q:** 250 bbl/hr (clearly infeasible — max achievable ≈ 182 bbl/hr)
- **Settled Q:** 175.0 bbl/hr (near theoretical maximum)
- **Ramp rate:** Never exceeded ±5%/step
- **Safety violations:** None
- **Behavior:** Controller ramped choke toward maximum as the infeasible target demanded, but correctly stopped short of 100% choke when safety constraints (FLP floor) were reached. It used the hard-limit-safe fallback when the conservative back-off band could not be satisfied, settling near ~86-87% choke with Q ≈ 175 bbl/hr — the maximum achievable within safe operating limits.

**PASS — correctly rejected infeasible target, settled at maximum safe rate.**

---

## Caveats and Limitations

1. **Identified tau vs true tau:** The identified model has tau = 3.52h (true simulator tau = 5.0h). This means the model predicts faster response than reality. The controller compensates by re-evaluating every step, but the model mismatch means the controller may be slightly less aggressive than optimal. In practice, this results in slower approach to target (conservative but safe).

2. **Safety margins add conservatism:** The 5 psi margins on WHP/FLP and 20 psi on BHP reduce the effective operating envelope. At high choke settings (>90%), this means the controller may not reach the absolute maximum Q because the safety back-off prevents exploring the boundary. This is intentional — robust control always trades peak performance for constraint satisfaction.

	To avoid deadlock when the measured state is safe but outside the conservative back-off envelope, the controller now falls back to a hard-limit-safe candidate search before giving up.

3. **No anti-windup or integral action:** The MPC uses a purely feedforward approach (no feedback correction for accumulated model error). Over very long runs, small systematic biases could accumulate. A production system would benefit from an integral term or state observer.

4. **Fixed horizon:** The 4-step MPC horizon is short. Longer horizons would better anticipate constraint violations during aggressive transients, but at higher computational cost. For this problem, 4 steps is sufficient given the 1-hour control interval and 5h time constant.

5. **Deterministic model:** The identified model is noise-free (no stochastic component). The MPC evaluates candidates deterministically, which is appropriate since the true simulator's noise is measurement-only and the underlying dynamics are deterministic.
