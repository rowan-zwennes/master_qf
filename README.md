# master_qf

Code for my MSc thesis in Quantitative Finance (Erasmus University Rotterdam, 2026):

**Optimal Market Making in Cryptocurrency Perpetual Futures: A Stochastic Control Extension for Discrete Funding Settlements and Nonlinear Alpha Signals**

Supervisor: Maria Grith

## What the thesis does

Market makers on perpetual futures pay or receive a funding payment on their open position at fixed times (every eight hours on Binance). Classical inventory models such as Guéant–Lehalle–Tapia (GLT) assume there is no such cost of carry.

The thesis:

- extends the GLT framework with an analytical correction for the discrete funding settlement. The payment is smoothed into a continuous drain in a short window before each settlement, so the ODE reduction of GLT is preserved, and the model sets the length of that window itself;
- adds a LightGBM drift signal trained on Level-2 order-book features (order-flow imbalance, book imbalance, VPIN, and others);
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
├── python/   data pipeline, HJB solvers, calibration, simulator, analyses
└── c++/      replay engine and latency benchmark
```

### Python

| Group | Files |
|---|---|
| Data recording | `recorder.py` (Binance WebSocket: book diffs, snapshots, trades, mark price) |
| Pipeline | `markprice_ffill.py` (stage 01), `book_reconstruct.py` (02), `feature_engineering.py` (03), `label_split.py` (04), `validate_*.py` (checks per stage), `data_gap_handler.py`, `pipeline_utils.py` |
| HJB / control | `hjb_asymptotic_reference.py` (Regime I closed form), `hjb_riccati_solver.py` (Regime II backward induction near settlement), `hjb_thomas_solver.py`, `hjb_epoch_stitching.py`, `hjb_regime_boundary.py`, `hjb_principal_eigenvector.py`, `hjb_lut_builder.py` |
| Calibration | `calibrate_intensity.py`, `calibrate_intensity_rolling.py`, `calibrate_volatility.py`, `calibrate_vpin.py`, `select_intensity_window.py`, `describe_rho.py` |
| Simulation | `run_simulation.py` (six-strategy replay), `sim_quote_engine.py` (quote rules), `fill_core_reference.py` (Python twin of the C++ fill core), `run_simulation_freshbook.py` |
| Stress tests | `stress_engine.py`, `stress_funding_ou.py`, `stress_q_sweep.py`, `fetch_altcoin_funding.py`, `rebuild_funding_table.py` |
| Analysis and figures | `analysis_*.py`, `fig_*.py`, `gen_results_tables.py`, `statistical_tests.py`, `run_statistical_tests.py` |

### C++

`hft_engine`: an event-by-event replay engine (order book, last-in-queue fill simulation, P&L tracking, GLT and funding-correction quotes from a precomputed lookup table) and a latency benchmark. `validate_cpp_parity.py` checks that it reproduces the Python simulator.

```
hft_engine replay --data day.hftr [--lut lut.bin] [--strategies 1,2,3,4] [--latency-out lat.hftb]
hft_engine bench  [--lut lut.bin] [--out lat.hftb]
```

## Requirements

Python 3.11+ and the packages in `Code Files/python/requirements.txt`:

```
pip install -r "Code Files/python/requirements.txt"
```

The C++ engine needs a C++17 compiler.

## Data

The raw order-book data (about 2 × 10⁸ trades and 8 × 10⁷ book updates, recorded 19 January – 23 April 2026) is not included because of its size. It can be re-recorded with `recorder.py`. Several scripts still point to the local data path used during the thesis (`/mnt/rowan_thuis/...`); change `DEFAULT_BASE` / `--base` to your own location.

## Citation

> Zwennes, R. (2026). *Optimal Market Making in Cryptocurrency Perpetual Futures: A Stochastic Control Extension for Discrete Funding Settlements and Nonlinear Alpha Signals.* MSc thesis, Erasmus School of Economics, Erasmus University Rotterdam.

A shortened version of the thesis is in preparation as a preprint.
