"""Battery (BESS) price-arbitrage optimization for ERCOT North Hub.

Perfect-foresight energy arbitrage LP, built once in Pyomo and solved with either
a CPU solver (HiGHS) or NVIDIA cuOpt (GPU), so the two can be compared fairly.

    max  sum_t  price[t] * (d[t] - c[t]) * dt  -  throughput_cost * (c[t] + d[t]) * dt
    s.t. soc[t] = soc[t-1] + dt * (eta_c * c[t] - d[t] / eta_d)
         0 <= c[t], d[t] <= available_mw[t]          (rated power if no telemetry)
         soc_min_frac * usable[t] <= soc[t] <= soc_max_frac * usable[t]
         soc[-1] = soc_init,  soc[T-1] >= soc_init   (no free energy at end of horizon)

`available_mw[t]` and `usable[t]` come from simulated telemetry (see telemetry.py):
derates, outages and capacity fade. Prices are real ERCOT HB_NORTH settlement point prices.

The LP has no binary "charge XOR discharge" variable. With the throughput cost and
round-trip losses, simultaneous charge/discharge is never profitable at positive
prices. It can show up at negative prices, so check the dispatch there if you care.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo


# --------------------------------------------------------------------------- #
# Battery spec
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BatterySpec:
    power_mw: float = 100.0
    energy_mwh: float = 200.0  # 2-hour battery
    eta_charge: float = 0.92  # ~0.85 round trip
    eta_discharge: float = 0.92
    soc_min_frac: float = 0.05
    soc_max_frac: float = 0.95
    soc_init_frac: float = 0.50
    throughput_cost: float = 2.0  # $/MWh of charge + discharge (degradation / VOM)


# --------------------------------------------------------------------------- #
# Price data
# --------------------------------------------------------------------------- #
def synthetic_north_hub_prices(
    n_days: int, interval_minutes: int = 15, seed: int = 0
) -> pd.Series:
    """ERCOT-North-Hub-*like* price series in $/MWh. NOT real data.

    Midday solar dip, evening ramp peak, day-to-day level shifts, AR(1) noise,
    occasional scarcity spikes and negative-price intervals.
    """
    rng = np.random.default_rng(seed)
    per_day = 24 * 60 // interval_minutes
    n = n_days * per_day
    hour = (np.arange(n) % per_day) * interval_minutes / 60.0

    shape = (
        28
        - 14 * np.exp(-0.5 * ((hour - 13.0) / 2.5) ** 2)  # solar dip
        + 38 * np.exp(-0.5 * ((hour - 19.5) / 1.8) ** 2)  # evening peak
        + 8 * np.exp(-0.5 * ((hour - 7.5) / 1.5) ** 2)  # morning shoulder
    )
    daily_level = np.repeat(rng.lognormal(0.0, 0.35, n_days), per_day)

    noise = np.zeros(n)
    eps = rng.normal(0, 5.0, n)
    for i in range(1, n):
        noise[i] = 0.85 * noise[i - 1] + eps[i]

    price = shape * daily_level + noise

    spikes = rng.random(n) < 0.002  # scarcity events
    price[spikes] += rng.exponential(400.0, spikes.sum())
    neg = (rng.random(n) < 0.01) & (hour > 9) & (hour < 16)
    price[neg] = -rng.exponential(12.0, neg.sum())

    price = np.clip(price, -250.0, 5000.0)
    idx = pd.date_range(
        "2024-01-01", periods=n, freq=f"{interval_minutes}min", tz="America/Chicago"
    )
    return pd.Series(price, index=idx, name="HB_NORTH_synthetic")


DATA_DIR = Path(__file__).parent / "data"
RT15_PATH = DATA_DIR / "hb_north_rt15.parquet"  # real ERCOT HB_NORTH, 15-min
DAM_PATH = DATA_DIR / "hb_north_dam.parquet"  # real ERCOT HB_NORTH, hourly
CONDITIONS_PATH = DATA_DIR / "bess_conditions_15min.parquet"  # SIMULATED, see telemetry.py


def load_prices_parquet(path: str | Path = RT15_PATH) -> pd.Series:
    """Real HB_NORTH settlement point prices ($/MWh) from the repo's Parquet files."""
    return pd.read_parquet(path)["price_usd_mwh"].rename("HB_NORTH")


def load_conditions_parquet(path: str | Path = CONDITIONS_PATH) -> pd.DataFrame:
    """SIMULATED battery conditions: ambient_temp_c, available_mw, usable_energy_mwh."""
    return pd.read_parquet(path)


def infer_dt_hours(prices: pd.Series) -> float:
    return float(pd.Series(prices.index).diff().dropna().median().total_seconds() / 3600.0)


# --------------------------------------------------------------------------- #
# Pyomo model
# --------------------------------------------------------------------------- #
def build_model(
    prices: np.ndarray,
    dt_hours: float,
    spec: BatterySpec = BatterySpec(),
    conditions: pd.DataFrame | None = None,
) -> pyo.ConcreteModel:
    """`conditions` (optional) must have one row per price interval with columns
    `available_mw` and `usable_energy_mwh`; without it the battery is always at
    rated power and nameplate energy."""
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    if conditions is None:
        p_max = np.full(n, spec.power_mw)
        usable = np.full(n, spec.energy_mwh)
    else:
        if len(conditions) != n:
            raise ValueError(f"conditions has {len(conditions)} rows, prices has {n}")
        p_max = conditions["available_mw"].to_numpy(dtype=float)
        usable = conditions["usable_energy_mwh"].to_numpy(dtype=float)
    e_min = spec.soc_min_frac * usable
    e_max = spec.soc_max_frac * usable
    e_init = spec.soc_init_frac * spec.energy_mwh

    m = pyo.ConcreteModel("bess_arbitrage")
    m.T = pyo.RangeSet(0, n - 1)
    m.charge = pyo.Var(m.T, bounds=lambda m, t: (0, p_max[t]))
    m.discharge = pyo.Var(m.T, bounds=lambda m, t: (0, p_max[t]))
    m.soc = pyo.Var(m.T, bounds=lambda m, t: (e_min[t], e_max[t]))

    def soc_balance(m, t):
        prev = e_init if t == 0 else m.soc[t - 1]
        return m.soc[t] == prev + dt_hours * (
            spec.eta_charge * m.charge[t] - m.discharge[t] / spec.eta_discharge
        )

    m.soc_balance = pyo.Constraint(m.T, rule=soc_balance)
    m.terminal_soc = pyo.Constraint(expr=m.soc[n - 1] >= e_init)

    m.profit = pyo.Objective(
        expr=sum(
            dt_hours
            * (
                prices[t] * (m.discharge[t] - m.charge[t])
                - spec.throughput_cost * (m.charge[t] + m.discharge[t])
            )
            for t in range(n)
        ),
        sense=pyo.maximize,
    )
    return m


def extract_dispatch(m: pyo.ConcreteModel) -> pd.DataFrame:
    """Dispatch in MW (positive = discharge) and SOC in MWh."""
    c = np.array([pyo.value(m.charge[t]) for t in m.T])
    d = np.array([pyo.value(m.discharge[t]) for t in m.T])
    s = np.array([pyo.value(m.soc[t]) for t in m.T])
    return pd.DataFrame({"charge_mw": c, "discharge_mw": d, "net_mw": d - c, "soc_mwh": s})


# --------------------------------------------------------------------------- #
# Solving
# --------------------------------------------------------------------------- #
CPU_SOLVER = "appsi_highs"
GPU_SOLVER = "cuopt"


def cuopt_available() -> bool:
    try:
        import cuopt  # noqa: F401

        return pyo.SolverFactory(GPU_SOLVER).available(exception_flag=False)
    except Exception:
        return False


def solve(m: pyo.ConcreteModel, solver_name: str, options: dict | None = None) -> dict:
    """Solve `m` in place. Returns objective, wall-clock seconds spent in
    `SolverFactory.solve` (includes Pyomo->solver transfer), and termination status."""
    solver = pyo.SolverFactory(solver_name)
    for k, v in (options or {}).items():
        solver.options[k] = v

    t0 = time.perf_counter()
    res = solver.solve(m)
    elapsed = time.perf_counter() - t0

    return {
        "solver": solver_name,
        "objective": pyo.value(m.profit),
        "solve_seconds": elapsed,
        "termination": str(res.solver.termination_condition),
    }


def warmup(solver_name: str, options: dict | None = None) -> None:
    """Tiny solve to pay one-time costs (CUDA context, library loading) up front,
    so they are not billed to the first benchmark size."""
    prices = synthetic_north_hub_prices(1, 60, seed=1).to_numpy()
    solve(build_model(prices, 1.0), solver_name, options)
