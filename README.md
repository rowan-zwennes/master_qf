# master_qf

Code for my MSc thesis in Quantitative Finance (Erasmus University Rotterdam, 2026):

**Optimal Market Making in Cryptocurrency Perpetual Futures: A Stochastic Control Extension for Discrete Funding Settlements and Nonlinear Alpha Signals**

Supervisor: Maria Grith

## What the thesis does

Market makers on perpetual futures pay or receive a funding payment on their open position at fixed times (every eight hours on Binance). Classical inventory models such as Guéant–Lehalle–Tapia (GLT) assume there is no such cost of carry.

The thesis:

- extends the GLT framework with an analytical correction for the discrete funding settlement. The payment is smoothed into a continuous drain in a short window before each settlement, so the ODE reduction of GLT is preserved, and the model sets the length of that window itself;
- adds a LightGBM drift signal trained on Level-2 order-book features (order-flow imbalance, book imbalance, VPIN, and others), compared against a Ridge baseline;
- evaluates six strategies in a fee-exact, queue-conservative replay of 94 days of self-recorded Binance BTCUSDT data, with a 64-day daily walk-forward out-of-sample period, plus a Monte Carlo stress suite with synthetic funding paths.

Main findings: at the retail fee tier, fees and adverse selection outweigh the captured spread for every strategy. The learned signal works mainly as a "risk clock" that predicts when a quote is about to be hit; used defensively, it cuts GLT's mean daily loss by 26% on 27% fewer fills. The funding correction activates where the theory predicts, and its value grows with the size of the funding rate.

## Strategies

| ID | Name | Drift | Funding correction |
|---|---|---|---|
| 1 | `naive_const_spread` | none | no (fixed half-spread, no GLT) |
| 2 | `glt_standard` | none | no |
| 3 | `glt_ar_drift` | rolling autoregression | no |
| 4 | `glt_ml_drift` | LightGBM | no |
| 5 | `glt_funding` | none | yes |
| 6 | `glt_full_hybrid` | LightGBM | yes |

## Repository layout

```
Code Files/
├── python/   data pipeline, machine learning, HJB solvers, calibration, simulator, analyses
└── c++/      replay engine and latency benchmark
```

### Python

| Group | Files |
|---|---|
| Data pipeline | `recorder.py` (Binance WebSocket: book diffs, snapshots, trades, mark price), `book_reconstruct.py` (100 ms top-20 book), `markprice_ffill.py`, `data_gap_handler.py`, `feature_engineering.py` (one-second feature grid), `label_split.py` (multi-horizon labels and train/test split), `pipeline_utils.py`, `validate_data.py`, `validate_book_reconstruct.py`, `validate_book.py`, `make_data_section_assets.py`, `trade_size_stats.py` |
| Machine learning | `ml_train.py` (walk-forward LightGBM with purged cross-validation and Optuna tuning), `ml_predict.py` (out-of-sample drift streams), `ml_baseline_ridge.py`, `compare_models.py` (Diebold–Mariano test), `compare_l2_huber.py`, `compute_purge_window.py`, `select_horizon.py`, `ml_crossfold_stability.py`, `gen_crossfold_table.py`, `ml_preanalysis_diagnostics.py`, `pre_analysis_tests.py`, `export_pre_analysis.py`, `ml_shap.py` (SHAP attributions), `check_directional_accuracy.py` |
| Calibration | `calibrate_intensity.py`, `calibrate_intensity_rolling.py` (fill intensity *(A, k)*), `select_intensity_window.py`, `validate_intensity_stride.py`, `validate_intensity_exp.py`, `calibrate_volatility.py`, `validate_sigma.py`, `describe_rho.py`, `calibrate_vpin.py` |
| HJB and quotes | `hjb_principal_eigenvector.py`, `hjb_riccati_solver.py` (backward induction near settlement), `hjb_thomas_solver.py`, `hjb_lut_builder.py`, `hjb_epoch_stitching.py`, `hjb_regime_boundary.py`, `hjb_asymptotic_reference.py` (closed-form GLT check), `sim_quote_engine.py` (quote rules for the six strategies), `quantify_ml_interp_error.py`, `analysis_decaying_drift.py`, `validate_gaussian_asymptotic.py`, `validate_cara_approx.py`, `validate_sref_approx.py`, `validate_boundary_layer.py` |
| Replay and stress tests | `run_simulation.py` (six-strategy replay with last-in-queue fills and exact fee and funding accounting), `fill_core_reference.py`, `run_simulation_freshbook.py`, `export_replay_binary.py`, `validate_cpp_parity.py`, `stress_engine.py`, `stress_funding_ou.py`, `stress_q_sweep.py`, `rebuild_funding_table.py`, `fetch_altcoin_funding.py` |
| Results and statistics | `statistical_tests.py`, `run_statistical_tests.py` (block bootstrap), `analysis_*.py`, `fig_*.py`, `sensitivity_common.py`, `gen_results_tables.py` |

### C++

`hft_engine` is an event-by-event replay engine: order book and queue position, last-in-queue fill simulation, GLT quotes with the funding correction from a precomputed lookup table, and P&L tracking. It also contains the latency benchmark. `validate_cpp_parity.py` checks fill by fill that it reproduces the Python simulator. `fill_core_runner` is a small test program for the fill core.

## Requirements

- Python 3.14 (developed on 3.14.2) and the packages in `Code Files/python/requirements.txt`:

  ```
  pip install -r "Code Files/python/requirements.txt"
  ```

- CMake 3.20+ and a C++20 compiler for the engine. It has no external dependencies.

## Building the C++ engine

```
cd "Code Files/c++"
cmake -S . -B build
cmake --build build -j
```

This produces `build/hft_engine` and `build/fill_core_runner`:

```
hft_engine replay --data day.hftr [--lut lut.bin] [--strategies 1,2,3,4] [--latency-out lat.hftb]
hft_engine bench  [--lut lut.bin] [--out lat.hftb]
```

The replay input (`.hftr`) is written by `export_replay_binary.py`, and the lookup table by `hjb_lut_builder.py`.

## Data

The raw order-book data (about 2 × 10⁸ trades and 8 × 10⁷ book updates, recorded 19 January – 23 April 2026) is not included because of its size. It can be re-recorded with `recorder.py`. Several scripts still point to the local data path used during the thesis (`/mnt/rowan_thuis/...`); change `DEFAULT_BASE` / `--base` to your own location.

## Citation

> Zwennes, R. (2026). *Optimal Market Making in Cryptocurrency Perpetual Futures: A Stochastic Control Extension for Discrete Funding Settlements and Nonlinear Alpha Signals.* MSc thesis, Erasmus School of Economics, Erasmus University Rotterdam.

A shortened version of the thesis is in preparation as a preprint.
