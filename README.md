# HoneyWell Autonomous Production Choke Controller

A hackathon submission for autonomous choke control of a single naturally flowing oil well.

## What it does

This project implements a complete control workflow:

1. Generates an open-loop step test across the full choke range.
2. Identifies a simple dynamic model from the step-test data.
3. Uses a brute-force MPC-style controller to select the next choke move.
4. Demonstrates three required closed-loop scenarios against the true simulator.

The controller respects:

- Choke bounds: 0% to 100%
- Choke ramp limit: +/-5% per control interval
- Active pressure constraints: WHP, FLP, BHP

If a target is infeasible, the controller settles at the maximum safe production rate.

## Core Files

- [simulator.py](simulator.py) - True simulator used as the process source
- [step_test_full.py](step_test_full.py) - Full-range open-loop step test generator
- [well_model.py](well_model.py) - Identified predictive model
- [controller.py](controller.py) - Brute-force MPC choke controller
- [run_scenarios.py](run_scenarios.py) - Scenario A/B/C closed-loop demonstration
- [generate_synthetic_dataset.py](generate_synthetic_dataset.py) - Synthetic data generator
- [config.py](config.py) - Safety limits and controller settings
- [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) - Results and narrative summary
- [tests/](tests/) - Validation suite

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the validation suite:

```bash
pytest -q
```

Generate the step-test data and plots:

```bash
python step_test_full.py
```

Run the closed-loop demonstration scenarios:

```bash
python run_scenarios.py
```

Generate the synthetic dataset:

```bash
python generate_synthetic_dataset.py --seed 42 --hours 400 --output synthetic_choke_dataset.csv
```

Generated CSV files live in `data/` and generated plots live in `images/`.

## Submission Highlights

- End-to-end process understanding through open-loop step testing
- Identified dynamic model with sub-1 bbl/hr RMSE on the primary production variable
- Safety-constrained controller that rejects unsafe candidates before selection
- Scenario coverage for startup, target tracking, and infeasible target handling
- Passing automated tests for simulator, model, and controller behavior

## Validation

The current test suite passes with 22 tests.
