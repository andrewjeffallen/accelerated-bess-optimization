"""Build the Parquet datasets used by the benchmark (written to data/).

    uv run python make_data.py fetch      # REAL prices from the ERCOT API (needs credentials)
    uv run python make_data.py telemetry  # SIMULATED battery telemetry (no credentials)

`fetch` needs ERCOT_USERNAME, ERCOT_PASSWORD and ERCOT_API_KEY in the environment.
`telemetry` reads data/hb_north_rt15.parquet, so anyone can regenerate it offline.

Files (all small enough to commit):
    hb_north_rt15.parquet             REAL  ERCOT HB_NORTH real-time 15-min SPP
    hb_north_dam.parquet              REAL  ERCOT HB_NORTH day-ahead hourly SPP
    bess_conditions_15min.parquet     SYNTHETIC  temperature, derates, outages, fade
    bess_telemetry_1min_sample.parquet SYNTHETIC  1-min SCADA-style telemetry for one week
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path(__file__).parent / "data"
CACHE_DIR = Path(__file__).parent / ".cache"  # raw API pulls, gitignored
RT_PATH = DATA_DIR / "hb_north_rt15.parquet"
DAM_PATH = DATA_DIR / "hb_north_dam.parquet"
CONDITIONS_PATH = DATA_DIR / "bess_conditions_15min.parquet"
TELEMETRY_PATH = DATA_DIR / "bess_telemetry_1min_sample.parquet"


def write_parquet(df: pd.DataFrame, path: Path, meta: dict[str, str]) -> None:
    """Write with provenance in the Parquet key-value metadata, so a file can't be
    mistaken for something it isn't (real vs simulated)."""
    table = pa.Table.from_pandas(df)
    merged = {**(table.schema.metadata or {}), **{k.encode(): v.encode() for k, v in meta.items()}}
    pq.write_table(table.replace_schema_metadata(merged), path, compression="zstd")
    print(f"wrote {path.relative_to(Path(__file__).parent)}  {len(df):,} rows  "
          f"{path.stat().st_size / 1024:,.0f} KiB")


def cmd_fetch(args) -> None:
    import ercot_api as ea

    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    DATA_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    client = ea.ERCOTClient()
    try:
        if not args.skip_rt:
            print("Fetching real-time 15-min SPP ...")
            rt_raw = ea.fetch_rt_spp(client, start, end)
            rt_raw.to_parquet(CACHE_DIR / "rt_raw.parquet")  # keep the raw pull if conversion fails
            rt = ea.rt_spp_to_frame(rt_raw)
            write_parquet(
                rt,
                RT_PATH,
                {"source": "ERCOT Public API np6-905-cd spp_node_zone_hub", "settlement_point": "HB_NORTH",
                 "synthetic": "false", "interval_minutes": "15", "tz": ea.TZ},
            )
        if not args.skip_dam:
            print("Fetching day-ahead hourly SPP ...")
            dam_raw = ea.fetch_dam_spp(client, start, end)
            dam_raw.to_parquet(CACHE_DIR / "dam_raw.parquet")
            dam = ea.dam_spp_to_frame(dam_raw)
            write_parquet(
                dam,
                DAM_PATH,
                {"source": "ERCOT Public API np4-190-cd dam_stlmnt_pnt_prices", "settlement_point": "HB_NORTH",
                 "synthetic": "false", "interval_minutes": "60", "tz": ea.TZ},
            )
    finally:
        client.close()


def cmd_telemetry(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import bess_opt as bo
    import telemetry as tl

    if not RT_PATH.exists():
        raise SystemExit(f"{RT_PATH} not found. Run `make_data.py fetch` first (or pull it from git).")
    spec = bo.BatterySpec()
    prices = bo.load_prices_parquet(RT_PATH)

    # 1) Simulated conditions for the whole price history (inputs to the optimizer).
    cond = tl.simulate_conditions(prices.index, spec, seed=args.seed)
    write_parquet(
        cond,
        CONDITIONS_PATH,
        {"synthetic": "true", "generator": "telemetry.simulate_conditions", "seed": str(args.seed),
         "note": "SIMULATED ambient temperature, derates, outages and capacity fade; not real plant data"},
    )

    # 2) Solve one sample week, then simulate what the plant would report while following it.
    n_week = 7 * 96
    i0 = prices.index.get_indexer([pd.Timestamp(args.sample_start, tz=prices.index.tz)])[0]
    if i0 < 0 or i0 + n_week > len(prices):
        raise SystemExit(f"--sample-start {args.sample_start} is outside the price history")
    p, c = prices.iloc[i0 : i0 + n_week], cond.iloc[i0 : i0 + n_week]
    m = bo.build_model(p.to_numpy(), bo.infer_dt_hours(p), spec, c)
    res = bo.solve(m, bo.CPU_SOLVER)
    print(f"sample week LP: {res['termination']}, profit ${res['objective']:,.0f}")
    disp = bo.extract_dispatch(m)

    net = pd.Series(disp["net_mw"].to_numpy(), index=p.index)
    tel = tl.simulate_realized_telemetry(net, cond, spec, seed=args.seed)
    write_parquet(
        tel,
        TELEMETRY_PATH,
        {"synthetic": "true", "generator": "telemetry.simulate_realized_telemetry", "seed": str(args.seed),
         "note": "SIMULATED 1-min SCADA-style telemetry following the optimal dispatch of the real-price LP"},
    )

    plan = pd.Series(100 * disp["soc_mwh"].to_numpy() / c["usable_energy_mwh"].to_numpy(),
                     index=p.index + pd.Timedelta(minutes=15))
    drift = (tel["soc_pct_true"].reindex(plan.index) - plan).abs()
    print(f"true SOC vs plan: mean |diff| {drift.mean():.2f} pct pts, max {drift.max():.2f}")
    Path("docs").mkdir(exist_ok=True)
    fig = tl.plot_telemetry(tel, plan, f"Simulated telemetry, week of {args.sample_start} (real HB_NORTH prices, synthetic plant)")
    fig.savefig("docs/telemetry_sample.png", dpi=130)
    print("wrote docs/telemetry_sample.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download real ERCOT prices")
    f.add_argument("--start", default="2023-12-11", help="earliest date the API returns real-time SPP")
    f.add_argument("--end", default="2026-11-19")
    f.add_argument("--skip-dam", action="store_true")
    f.add_argument("--skip-rt", action="store_true", help="only (re)fetch day-ahead prices")
    f.set_defaults(func=cmd_fetch)

    t = sub.add_parser("telemetry", help="simulate battery telemetry from the price file")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--sample-start", default="2025-07-14", help="first day of the 1-min sample week")
    t.set_defaults(func=cmd_telemetry)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
