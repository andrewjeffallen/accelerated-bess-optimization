# accelerated-bess-optimization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andrewjeffallen/accelerated-bess-optimization/blob/main/bess_cuopt_benchmark.ipynb)

GPU-accelerated BESS optimization test & benchmark. A battery price-arbitrage LP for
**ERCOT North Hub (`HB_NORTH`)**, modeled once in Pyomo and solved two ways:

| | Solver | Hardware |
|---|---|---|
| Baseline | Pyomo + HiGHS (`appsi_highs`) | CPU |
| Accelerated | Pyomo + NVIDIA cuOpt (`cuopt`) | GPU |

## Run it in Google Colab

1. Click the **Open in Colab** badge above (or open `bess_cuopt_benchmark.ipynb` via `File -> Open notebook -> GitHub`).
2. `Runtime -> Change runtime type -> GPU` (T4 is fine).
3. Run all cells. The notebook installs dependencies and clones this repo, so the real price data comes
   with it and no credentials are needed.

The notebook defaults to the real HB_NORTH prices plus simulated battery conditions. Set
`SOURCE = "synthetic"` in the data cell to reproduce the earlier synthetic-price runs.

If you restart the session after installing, re-run from the top.

## Run it locally (CPU only)

This project uses [uv](https://docs.astral.sh/uv/) for package management.

```bash
uv sync
uv run python benchmark.py --cpu-only --days 1 7 30 90
```

By default this uses the real prices and simulated conditions in `data/`. Useful flags:
`--no-conditions` (battery always at rated limits), `--source synthetic` (generated prices of any
length, see `--interval-minutes`), `--repeats N`, `--out results.csv`.

On a Linux machine with an NVIDIA GPU (CUDA 12 or 13, Python 3.11+), sync with the matching
cuOpt extra and drop `--cpu-only`:

```bash
uv sync --extra cuda12       # CUDA 12.x   (or: uv sync --extra cuda13)
uv run python benchmark.py --days 1 7 30 90 365 730 1074
```

## The model

Perfect-foresight energy arbitrage for one battery (default 100 MW / 200 MWh, 92% one-way
efficiency, $2/MWh throughput cost). Variables per interval: charge, discharge, state of charge.
Constraints: SOC balance, power/SOC bounds, and end-of-horizon SOC >= start SOC. Pure LP, no binaries.

Size grows linearly with the horizon: `3 * intervals` variables and `intervals + 1` constraints
(about 105k variables for a year of 15-minute data).

## Dataset overview

All files are small Parquet files in `data/` (about 2.5 MB total), committed to the repo so Colab and
local runs need no credentials. Each file carries a `synthetic` flag and its provenance in the Parquet
key-value metadata, so real and simulated data can't be mixed up.

| File | Real or simulated | Rows | Used by the LP? |
|---|---|---|---|
| `hb_north_rt15.parquet` | **Real**: ERCOT real-time settlement point prices, 15-min | 103,143 | **Yes**: the price signal |
| `bess_conditions_15min.parquet` | **Simulated** | 103,143 | Optional: time-varying power and energy limits |
| `hb_north_dam.parquet` | **Real**: ERCOT day-ahead settlement point prices, hourly | 25,752 | No: available for day-ahead vs real-time comparisons |
| `bess_telemetry_1min_sample.parquet` | **Simulated** | 10,080 | No: a demo of what plant telemetry could look like (docs/telemetry_sample.png) |

The real-time prices and the conditions cover 2023-12-11 14:15 to 2026-11-19 23:45 (1,074 full days;
the API returns no real-time prices before 2023-12-11). The day-ahead prices cover 2023-12-13 00:00 to
2026-11-19 23:00. All files are indexed by interval start and use tz-aware `America/Chicago` timestamps.

### Real data: ERCOT prices

- **Source:** [ERCOT Public API](https://developer.ercot.com/), settlement point `HB_NORTH` (North Hub).
  - Real-time 15-min: report `np6-905-cd` (`spp_node_zone_hub`)
  - Day-ahead hourly: report `np4-190-cd` (`dam_stlmnt_pnt_prices`)
- **Columns:** `price_usd_mwh` (in $/MWh) and `is_filled` (true if the interval was missing in the
  source and linearly interpolated).
- **Cleaning** (`ercot_api.py`): ERCOT's local interval times are converted to tz-aware timestamps using
  its `DSTFlag` (so the repeated fall-back hour is handled), then put on a complete regular grid.
  Gaps of up to 60 minutes are interpolated and flagged. Longer gaps raise an error. Rows the API repeats
  verbatim are dropped (a repeat with a different price still raises). The real-time file has 4 filled
  intervals out of 103,143 and the day-ahead file has none. There are no nulls or duplicate timestamps.
- **Values:** real-time prices run from about -$72 to $4,965/MWh (mean about $30). Day-ahead prices run
  from -$6 to $2,225/MWh (mean $31). Spikes and negative prices are real and are what the battery arbitrages.
- **How it is used:** price is the only objective coefficient. The LP maximizes
  `sum(price[t] * (discharge[t] - charge[t]) * dt)` minus the throughput cost. The time step `dt` is
  inferred from the price index (0.25 h), so the price file also sets the problem size
  (`3 * rows` variables, about 309k for the full 1,074 days).

### Simulated data: battery conditions

No real plant data is used. `telemetry.py` generates seeded, reproducible conditions (seed 0) so the
optimizer has realistic time-varying limits. **Nothing here is measured from a real battery.**

`bess_conditions_15min.parquet`, aligned row for row with the real-time prices:

| Column | Meaning | Used by the LP? |
|---|---|---|
| `available_mw` | Power limit for charge and discharge: rated 100 MW times a thermal derate (up to 15% in heat, 10% in cold) times partial or full outages (about 18 per year, median 6 h) | Yes: upper bound on `charge[t]` and `discharge[t]` |
| `usable_energy_mwh` | Energy capacity with linear fade of 2.5% per year, from 200 to about 190 MWh | Yes: scales the SOC bounds |
| `ambient_temp_c` | North-Texas-like temperature (seasonal + daily + multi-day weather), not real weather | No: drives the derates and the telemetry sample |

`bess_telemetry_1min_sample.parquet` is one week (2025-07-14) of 1-minute SCADA-style telemetry (setpoint
vs actual MW, true vs BMS-reported SOC, cell temperature, DC voltage). It is produced by solving the
LP on that week of real prices and then simulating a plant that follows the dispatch imperfectly. It is
an output for illustration, not an input to the benchmark.

### Synthetic prices

`bess_opt.synthetic_north_hub_prices(n_days, interval_minutes, seed)` generates an ERCOT-like price
series (solar dip, evening peak, spikes, negative intervals). **It is not real data.** It is opt-in
(`--source synthetic` in `benchmark.py`, `SOURCE = "synthetic"` in the notebook) for horizons longer than
the 1,074 days of real data or for other time steps. Synthetic runs use no battery conditions.

### Other model inputs

The battery itself (`BatterySpec` in `bess_opt.py`) is configuration, not data: 100 MW / 200 MWh,
92% one-way efficiency, SOC limits of 5-95%, 50% initial SOC, and $2/MWh throughput cost. Without a
conditions file the battery is always at rated power and nameplate energy.

### Loading and regenerating

```python
import bess_opt as bo
prices = bo.load_prices_parquet()          # real HB_NORTH, 15-min
conditions = bo.load_conditions_parquet()  # simulated limits, same index as prices
m = bo.build_model(prices.to_numpy(), bo.infer_dt_hours(prices), conditions=conditions)
```

`build_model` requires `conditions` to have exactly one row per price interval, so slice both to the
same range. The benchmark takes the first N days of both. The conditions file was generated for the
default 100 MW / 200 MWh battery, so if you change `BatterySpec`, regenerate it with
`make_data.py telemetry` or run with `--no-conditions`.

```bash
uv run python make_data.py telemetry   # rebuild simulated files from the price file, offline
# Re-download real prices (needs ERCOT_USERNAME, ERCOT_PASSWORD, ERCOT_API_KEY in the environment):
uv run python make_data.py fetch --start 2023-12-11 --end 2026-11-19
```

### Caveats

- Prices are real, but a perfect-foresight LP knows every future price and the battery is a price
  taker, so profits are an upper bound and not a forecast. For benchmarking solver speed this doesn't matter.
- Battery conditions are simulated and illustrative. Real derates, outages and degradation will differ.
- Check ERCOT's terms of use before redistributing the price files beyond this repo.

## What to expect

HiGHS is very good on LPs of this shape and will usually win on small and mid-sized
problems. cuOpt's GPU method (PDLP) targets large LPs, so the interesting question is where, or whether,
the curves cross as the horizon grows. GPU first-order methods also stop at a tolerance, so the
benchmark reports the objective gap between solvers. Compare timings only when that gap is small.

Timing covers `SolverFactory.solve(...)` (including Pyomo-to-solver transfer). Pyomo model-build
time is the same for both and is reported separately.

## Files

- `bess_opt.py`: price loaders, Pyomo model, solve helpers
- `benchmark.py`: sweep, summary table, plots, CLI
- `bess_cuopt_benchmark.ipynb`: Colab notebook
- `data/`: Parquet datasets (see [Dataset overview](#dataset-overview))
- `make_data.py`: builds `data/` (`fetch` for real ERCOT prices, `telemetry` for simulated data)
- `ercot_api.py`: minimal ERCOT Public API client and price cleaning
- `telemetry.py`: simulated battery conditions and 1-minute telemetry
- `docs/`: telemetry plots
- `pyproject.toml` / `uv.lock`: dependencies. cuOpt is an optional extra (`cuda12` / `cuda13`) because it needs a GPU and Linux.

## Notes

- cuOpt's Pyomo plugin requires a recent Pyomo (`>=6.10`).
- Extra cuOpt solver settings can be passed via `GPU_OPTIONS` in the notebook (forwarded to cuOpt's
  `SolverSettings.set_parameter`).
