"""Benchmark: Pyomo + HiGHS (CPU) vs Pyomo + NVIDIA cuOpt (GPU) on the BESS arbitrage LP.

    python benchmark.py --source synthetic --days 1 7 30 90 365 --interval-minutes 15
    python benchmark.py --source ercot --start 2024-01-01 --end 2024-12-31
    python benchmark.py --cpu-only          # no GPU needed
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import bess_opt as bo


def run_benchmark(
    prices: pd.Series,
    day_counts: list[int],
    spec: bo.BatterySpec = bo.BatterySpec(),
    cpu_options: dict | None = None,
    gpu_options: dict | None = None,
    run_gpu: bool = True,
    repeats: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    """Solve the first N days of `prices` for each N in `day_counts` on each solver."""
    dt = bo.infer_dt_hours(prices)
    per_day = int(round(24 / dt))
    solvers = [(bo.CPU_SOLVER, cpu_options)]
    if run_gpu:
        solvers.append((bo.GPU_SOLVER, gpu_options))

    for name, opts in solvers:  # pay one-time init costs before timing anything
        bo.warmup(name, opts)

    rows = []
    for days in day_counts:
        n = days * per_day
        if n > len(prices):
            if verbose:
                print(f"skip {days}d: only {len(prices) / per_day:.0f} days of prices")
            continue
        p = prices.iloc[:n].to_numpy()

        t0 = time.perf_counter()
        probe = bo.build_model(p, dt, spec)
        build_s = time.perf_counter() - t0
        n_vars = 3 * n
        n_cons = n + 1

        for name, opts in solvers:
            best = None
            for _ in range(repeats):
                m = bo.build_model(p, dt, spec)  # fresh model: no warm-start reuse
                try:
                    r = bo.solve(m, name, opts)
                except Exception as exc:  # keep the sweep going if one solver fails
                    r = {"solver": name, "objective": np.nan, "solve_seconds": np.nan,
                         "termination": f"error: {exc}"}
                if best is None or r["solve_seconds"] < best["solve_seconds"]:
                    best = r
            rows.append({"days": days, "intervals": n, "vars": n_vars, "cons": n_cons,
                         "pyomo_build_s": build_s, **best})
            if verbose:
                print(f"{days:>4}d {n:>7} intervals  {name:<12} "
                      f"{best['solve_seconds']:8.3f}s  obj={best['objective']:>14,.2f}  "
                      f"[{best['termination']}]")
        del probe

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """One row per problem size: both solve times, speedup, and objective agreement."""
    cpu = df[df.solver == bo.CPU_SOLVER].set_index("days")
    out = cpu[["intervals", "vars", "pyomo_build_s", "solve_seconds", "objective"]].rename(
        columns={"solve_seconds": "cpu_s", "objective": "cpu_obj"}
    )
    gpu = df[df.solver == bo.GPU_SOLVER].set_index("days")
    if len(gpu):
        out["gpu_s"] = gpu["solve_seconds"]
        out["gpu_obj"] = gpu["objective"]
        out["speedup_x"] = out["cpu_s"] / out["gpu_s"]  # >1 means GPU faster
        out["obj_rel_gap"] = (out["gpu_obj"] - out["cpu_obj"]).abs() / out["cpu_obj"].abs().clip(lower=1e-9)
    return out.reset_index()


def plot_results(summary: pd.DataFrame, save_path: str | None = None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2 if "gpu_s" in summary else 1, figsize=(12, 4.5), squeeze=False)
    ax = ax[0]
    ax[0].plot(summary["vars"], summary["cpu_s"], "o-", label="Pyomo + HiGHS (CPU)")
    if "gpu_s" in summary:
        ax[0].plot(summary["vars"], summary["gpu_s"], "s-", label="Pyomo + cuOpt (GPU)")
    ax[0].set(xscale="log", yscale="log", xlabel="LP variables", ylabel="solve time (s)",
              title="Solve time vs problem size")
    ax[0].grid(True, which="both", alpha=0.3)
    ax[0].legend()

    if "gpu_s" in summary:
        ax[1].bar(summary["days"].astype(str), summary["speedup_x"])
        ax[1].axhline(1.0, color="k", lw=1, ls="--")
        ax[1].set(xlabel="horizon (days)", ylabel="speedup  (CPU time / GPU time)",
                  title="GPU speedup (>1 = GPU faster)", yscale="log")
        ax[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_dispatch(prices: pd.Series, dispatch: pd.DataFrame, title: str = "Optimal dispatch"):
    import matplotlib.pyplot as plt

    idx = prices.index[: len(dispatch)]
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(idx, prices.iloc[: len(dispatch)].to_numpy(), lw=1)
    axes[0].set_ylabel("$/MWh")
    axes[0].set_title(title)
    axes[1].fill_between(idx, dispatch["net_mw"], step="mid", alpha=0.7)
    axes[1].set_ylabel("net MW (+ discharge)")
    axes[2].plot(idx, dispatch["soc_mwh"], lw=1)
    axes[2].set_ylabel("SOC (MWh)")
    fig.tight_layout()
    return fig


def load_prices(args) -> pd.Series:
    max_days = max(args.days)
    if args.source == "ercot":
        try:
            return bo.load_ercot_north_hub(args.start, args.end, args.market)
        except Exception as exc:
            print(f"[warn] could not load ERCOT prices ({exc}); using SYNTHETIC prices instead")
    elif args.source == "csv":
        return bo.load_prices_csv(args.csv, args.time_col, args.price_col)
    return bo.synthetic_north_hub_prices(max_days, args.interval_minutes, seed=args.seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["synthetic", "ercot", "csv"], default="synthetic")
    ap.add_argument("--days", type=int, nargs="+", default=[1, 7, 30, 90, 365])
    ap.add_argument("--interval-minutes", type=int, default=15, help="synthetic data only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--market", choices=["RT15", "DAM"], default="RT15")
    ap.add_argument("--csv")
    ap.add_argument("--time-col", default="timestamp")
    ap.add_argument("--price-col", default="price")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--out", default="benchmark_results.csv")
    args = ap.parse_args()

    run_gpu = not args.cpu_only
    if run_gpu and not bo.cuopt_available():
        print("[warn] cuOpt not available (no GPU / not installed); running CPU only")
        run_gpu = False

    prices = load_prices(args)
    print(f"{len(prices)} price intervals, dt={bo.infer_dt_hours(prices) * 60:.0f} min\n")
    df = run_benchmark(prices, args.days, run_gpu=run_gpu, repeats=args.repeats)
    summary = summarize(df)
    print("\n", summary.to_string(index=False, float_format=lambda x: f"{x:,.4g}"))
    summary.to_csv(args.out, index=False)
    plot_results(summary, save_path=args.out.replace(".csv", ".png"))


if __name__ == "__main__":
    main()
