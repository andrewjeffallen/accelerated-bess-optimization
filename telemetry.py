"""SIMULATED battery telemetry. Nothing in this module is real plant data.

Two pieces, both seeded and reproducible:

1. `simulate_conditions`  (15-min, aligned to the price index)
   What the plant could do in each interval: ambient temperature, available power
   (heat/cold derates + random partial/full outages) and usable energy (capacity
   fade). These feed the optimizer as time-varying limits.

2. `simulate_realized_telemetry`  (1-min, SCADA-style)
   What a plant following the optimizer's dispatch would report: setpoint vs actual
   MW (ramp lag + noise), true vs BMS-reported SOC (sensor bias/noise), cell
   temperature, DC voltage. The "plant" deliberately differs from the optimizer's
   model (load/temperature-dependent efficiency, curtailment at SOC limits), so the
   realized SOC drifts away from the plan, as it would open-loop on a real asset.

Run `python telemetry.py` to simulate a week and save a plot to docs/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bess_opt import BatterySpec


def _ar1(n: int, phi: float, sigma_stationary: float, rng: np.random.Generator) -> np.ndarray:
    """AR(1) with the given lag-1 coefficient and stationary standard deviation."""
    eps = rng.normal(0.0, sigma_stationary * np.sqrt(1 - phi**2), n)
    x = np.empty(n)
    x[0] = rng.normal(0.0, sigma_stationary)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


# --------------------------------------------------------------------------- #
# 1. Conditions (15-min inputs to the optimizer)
# --------------------------------------------------------------------------- #
def simulate_conditions(
    index: pd.DatetimeIndex,
    spec: BatterySpec = BatterySpec(),
    seed: int = 0,
    outages_per_year: float = 18.0,
    annual_fade: float = 0.025,
) -> pd.DataFrame:
    """Per-interval ambient temp, available power and usable energy.

    Ambient is North-Texas-like (seasonal + diurnal + multi-day weather), not real weather.
    Available power = rated * thermal derate * (1 - outage loss).
    """
    rng = np.random.default_rng(seed)
    n = len(index)
    step_h = float(index.to_series().diff().dropna().median().total_seconds() / 3600)
    local = index.tz_convert("America/Chicago")
    hour = local.hour.to_numpy() + local.minute.to_numpy() / 60.0
    doy = local.dayofyear.to_numpy()

    seasonal = 19.0 + 12.0 * np.sin(2 * np.pi * (doy - 110) / 365.25)  # peaks ~20 Jul
    diurnal = 6.0 * np.sin(2 * np.pi * (hour - 9.0) / 24.0)  # peaks ~15:00
    weather = _ar1(n, phi=np.exp(-step_h / 36.0), sigma_stationary=4.0, rng=rng)  # ~1.5 day memory
    ambient = seasonal + diurnal + weather

    # Thermal derate: up to 15% at 45C, up to 10% at -10C.
    hot = np.clip((ambient - 38.0) / 7.0, 0, 1) * 0.15
    cold = np.clip((0.0 - ambient) / 10.0, 0, 1) * 0.10
    thermal_frac = 1.0 - np.maximum(hot, cold)

    # Outages: Poisson arrivals, lognormal duration (median 6h), lose 25/50/100% of blocks.
    loss = np.zeros(n)
    n_events = rng.poisson(outages_per_year * n * step_h / (365.25 * 24))
    for _ in range(n_events):
        i0 = rng.integers(0, n)
        dur = max(1, int(round(rng.lognormal(np.log(6.0), 0.8) / step_h)))
        sev = rng.choice([0.25, 0.5, 1.0], p=[0.5, 0.3, 0.2])
        loss[i0 : i0 + dur] = np.maximum(loss[i0 : i0 + dur], sev)

    years = (index - index[0]).total_seconds().to_numpy() / (365.25 * 86400)
    usable = spec.energy_mwh * (1.0 - annual_fade * years)

    return pd.DataFrame(
        {
            "ambient_temp_c": ambient.round(2),
            "available_mw": (spec.power_mw * thermal_frac * (1.0 - loss)).round(3),
            "usable_energy_mwh": usable.round(3),
        },
        index=index,
    )


# --------------------------------------------------------------------------- #
# 2. Realized 1-minute telemetry
# --------------------------------------------------------------------------- #
def _cell_ocv(soc: float) -> float:
    """Open-circuit voltage per cell (V): LFP-like flat plateau with steep ends."""
    return 3.25 + 0.10 * (soc - 0.5) - 0.25 * np.exp(-soc / 0.05) + 0.20 * np.exp(-(1 - soc) / 0.04)


def simulate_realized_telemetry(
    net_mw: pd.Series,
    conditions: pd.DataFrame,
    spec: BatterySpec = BatterySpec(),
    soc_start_mwh: float | None = None,
    seed: int = 0,
    n_series_cells: int = 416,
) -> pd.DataFrame:
    """1-minute telemetry for a plant following `net_mw` (15-min setpoints, + = discharge).

    `soc_pct` is what the BMS reports (bias + noise); `soc_pct_true` is the simulated
    ground truth. `dc_voltage_v` is a rack-equivalent pack voltage.
    """
    rng = np.random.default_rng(seed)
    end = net_mw.index[-1] + (net_mw.index[1] - net_mw.index[0])
    idx = pd.date_range(net_mw.index[0], end, freq="1min", inclusive="left")
    setpoint = net_mw.reindex(idx, method="ffill").to_numpy()
    cond = conditions.reindex(idx, method="ffill")
    avail = cond["available_mw"].to_numpy()
    usable = cond["usable_energy_mwh"].to_numpy()
    ambient = cond["ambient_temp_c"].to_numpy()
    n = len(idx)
    dt_h = 1.0 / 60.0

    ramp = 0.5 * spec.power_mw  # MW per minute: full swing in ~2 min
    p_noise = _ar1(n, 0.7, 0.004 * spec.power_mw, rng)
    bias = _ar1(n, 1 - 1 / 2880, 0.008, rng)  # BMS SOC bias, ~2-day memory, ~0.8% std
    soc_noise = rng.normal(0.0, 0.001, n)
    v_noise = rng.normal(0.0, 0.5, n)

    soc_mwh = spec.soc_init_frac * spec.energy_mwh if soc_start_mwh is None else soc_start_mwh
    t_cell = float(ambient[0]) + 3.0
    p_prev = 0.0

    actual = np.empty(n)
    soc_true = np.empty(n)
    t_cells = np.empty(n)
    volts = np.empty(n)

    for k in range(n):
        # Command tracking: clip to what is available, ramp-limit, add noise.
        cmd = float(np.clip(setpoint[k], -avail[k], avail[k]))
        p = p_prev + float(np.clip(cmd - p_prev, -ramp, ramp)) + p_noise[k]
        p = float(np.clip(p, -avail[k], avail[k]))

        # Plant efficiency is worse than the optimizer's constant: it drops with
        # power level and in the cold.
        frac = abs(p) / spec.power_mw
        eta_c = spec.eta_charge - 0.010 * frac**2 - 0.0005 * max(0.0, 15.0 - t_cell)
        eta_d = spec.eta_discharge - 0.010 * frac**2 - 0.0005 * max(0.0, 15.0 - t_cell)

        # BMS protects the cells: curtail power that would leave [2%, 98%] of usable energy.
        lo, hi = 0.02 * usable[k], 0.98 * usable[k]
        if p > 0:
            p = min(p, max(0.0, (soc_mwh - lo)) * eta_d / dt_h)
        elif p < 0:
            p = max(p, -max(0.0, (hi - soc_mwh)) / (eta_c * dt_h))

        if p >= 0:
            soc_mwh -= dt_h * p / eta_d
            loss_mw = p * (1 / eta_d - 1)
        else:
            soc_mwh += dt_h * (-p) * eta_c
            loss_mw = -p * (1 - eta_c)
        soc_mwh = float(np.clip(soc_mwh, 0.0, usable[k]))

        # First-order thermal model: conversion losses heat the cells; HVAC pulls the
        # surroundings toward 25C when it is warmer outside.
        sink = ambient[k] + 0.4 * (25.0 - ambient[k]) if ambient[k] > 25.0 else ambient[k]
        t_cell += dt_h * (1.67 * loss_mw - 1.33 * (t_cell - sink))

        soc_frac = soc_mwh / usable[k]
        c_rate = p / spec.energy_mwh  # + discharging
        volts[k] = n_series_cells * (_cell_ocv(soc_frac) - 0.05 * c_rate / 0.5) + v_noise[k]

        actual[k], soc_true[k], t_cells[k], p_prev = p, soc_frac, t_cell, p

    soc_bms = np.clip(soc_true + bias + soc_noise, 0.0, 1.0)
    return pd.DataFrame(
        {
            "setpoint_mw": setpoint.round(3),
            "actual_mw": actual.round(3),
            "soc_pct": (100 * soc_bms).round(3),
            "soc_pct_true": (100 * soc_true).round(3),
            "cell_temp_c": t_cells.round(2),
            "ambient_temp_c": ambient.round(2),
            "dc_voltage_v": volts.round(1),
            "available_mw": avail.round(3),
            "online": avail > 0,
        },
        index=idx,
    )


def plot_telemetry(tel: pd.DataFrame, planned_soc_pct: pd.Series | None = None, title: str = ""):
    """Four stacked panels: power, SOC, temperature, voltage."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    ax[0].plot(tel.index, tel["setpoint_mw"], lw=0.8, label="setpoint (optimizer)")
    ax[0].plot(tel.index, tel["actual_mw"], lw=0.8, label="actual (telemetry)")
    ax[0].plot(tel.index, tel["available_mw"], "k:", lw=0.8, label="available")
    ax[0].plot(tel.index, -tel["available_mw"], "k:", lw=0.8)
    ax[0].set_ylabel("MW (+ discharge)")
    ax[0].legend(ncol=3, fontsize=8, loc="upper right")
    ax[0].set_title(title or "Simulated BESS telemetry (synthetic)")

    if planned_soc_pct is not None:
        ax[1].step(planned_soc_pct.index, planned_soc_pct, where="post", lw=1, alpha=0.6, label="optimizer plan")
    ax[1].plot(tel.index, tel["soc_pct_true"], lw=0.9, label="true SOC")
    ax[1].plot(tel.index, tel["soc_pct"], lw=0.6, alpha=0.8, label="BMS-reported SOC")
    ax[1].set_ylabel("SOC (%)")
    ax[1].legend(ncol=3, fontsize=8, loc="upper right")

    ax[2].plot(tel.index, tel["cell_temp_c"], lw=0.9, label="cell")
    ax[2].plot(tel.index, tel["ambient_temp_c"], lw=0.9, label="ambient")
    ax[2].set_ylabel("deg C")
    ax[2].legend(ncol=2, fontsize=8, loc="upper right")

    ax[3].plot(tel.index, tel["dc_voltage_v"], lw=0.6)
    ax[3].set_ylabel("DC voltage (V)")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # Standalone demo: one synthetic week driven by a trivial rule-based schedule.
    import matplotlib

    matplotlib.use("Agg")
    import pathlib

    from bess_opt import synthetic_north_hub_prices

    spec = BatterySpec()
    prices = synthetic_north_hub_prices(7, 15, seed=0)
    cond = simulate_conditions(prices.index.tz_convert("UTC"), spec, seed=0)
    hour = prices.index.hour + prices.index.minute / 60
    rule = pd.Series(
        np.where((hour >= 11) & (hour < 15), -spec.power_mw, np.where((hour >= 18) & (hour < 22), spec.power_mw, 0.0)),
        index=cond.index,
    )
    tel = simulate_realized_telemetry(rule, cond, spec)
    out = pathlib.Path("docs")
    out.mkdir(exist_ok=True)
    plot_telemetry(tel, title="Demo: rule-based schedule on synthetic prices").savefig(out / "telemetry_demo.png", dpi=130)
    print(tel.describe().T[["min", "mean", "max"]].round(2))
