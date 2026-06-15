"""Per-method tuning sweep signal: range/seedSD and seed-blocked ANOVA from results.sqlite."""

from __future__ import annotations

import argparse
import logging
import math
import sqlite3
from typing import Any

from scripts.analysis.results.core import connect, init_schema
from scripts.common import ensure_dirs, load_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument(
        "--benchmark",
        default="all",
        choices=["patch_feature", "wsi_bag", "all"],
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Minimum range/seedSD ratio to declare a sweep informative.",
    )
    return p.parse_args()


def _load_tuning_f1(
    conn: sqlite3.Connection, benchmarks: list[str]
) -> list[dict[str, Any]]:
    placeholders = ",".join("?" * len(benchmarks))
    rows = conn.execute(
        f"""
        SELECT r.benchmark, r.method, r.seed, r.tuning_params_json, e.macro_f1
        FROM runs r
        JOIN eval_results e ON r.run_id = e.run_id
        WHERE e.split = 'val'
          AND r.tuning_params_json IS NOT NULL
          AND r.tuning_params_json != '{{}}'
          AND r.benchmark IN ({placeholders})
        ORDER BY r.benchmark, r.method, r.tuning_params_json, r.seed
        """,
        benchmarks,
    ).fetchall()
    return [
        {"benchmark": bm, "method": m, "seed": s, "params": q, "val_macro_f1": f}
        for bm, m, s, q, f in rows
    ]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _variance(xs: list[float]) -> float:
    mu = _mean(xs)
    return sum((x - mu) ** 2 for x in xs) / (len(xs) - 1) if len(xs) >= 2 else 0.0


def _pooled_seed_sd(groups: list[list[float]]) -> float:
    """Root of pooled within-group variance across configs."""
    total_ss, total_df = 0.0, 0
    for g in groups:
        if len(g) >= 2:
            total_ss += _variance(g) * (len(g) - 1)
            total_df += len(g) - 1
    return math.sqrt(total_ss / total_df) if total_df > 0 else float("nan")


def _betai(a: float, b: float, x: float, iterations: int = 200) -> float:
    """Regularized incomplete beta function via continued fraction (Lentz)."""
    if x < 0.0 or x > 1.0:
        raise ValueError
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betai(b, a, 1.0 - x, iterations)
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - log_beta) / a
    tiny = 1e-300
    f, c, d = tiny, tiny, 0.0
    for m in range(iterations):
        for step in (0, 1):
            if m == 0 and step == 0:
                num = 1.0
            elif step == 0:
                num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
            else:
                num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
            d = 1.0 / max(abs(1.0 + num * d), tiny)
            c = max(abs(1.0 + num / c), tiny)
            delta = c * d
            f *= delta
            if abs(delta - 1.0) < 1e-10:
                break
    return front * f


def _f_pvalue(f_stat: float, df1: int, df2: int) -> float:
    """Survival function of the F distribution via regularized incomplete beta."""
    if f_stat <= 0.0:
        return 1.0
    return _betai(df2 / 2.0, df1 / 2.0, df2 / (df2 + df1 * f_stat))


def _anova_two_way(
    configs: list[str], seeds: list[int], values: dict[tuple[str, int], float]
) -> tuple[float, float, int, int]:
    """Two-way ANOVA with seed blocked; returns (F, p, df_config, df_error)."""
    n_c, n_s = len(configs), len(seeds)
    grand = _mean(list(values.values()))
    cm = {c: _mean([values[c, s] for s in seeds if (c, s) in values]) for c in configs}
    sm = {s: _mean([values[c, s] for c in configs if (c, s) in values]) for s in seeds}
    ss_c = n_s * sum((cm[c] - grand) ** 2 for c in configs)
    ss_s = n_c * sum((sm[s] - grand) ** 2 for s in seeds)
    ss_err = sum((v - grand) ** 2 for v in values.values()) - ss_c - ss_s
    df_c, df_err = n_c - 1, (n_c - 1) * (n_s - 1)
    if df_err <= 0 or ss_err <= 0:
        return float("nan"), float("nan"), df_c, df_err
    ms_c = ss_c / df_c if df_c > 0 else 0.0
    ms_err = ss_err / df_err
    if ms_err == 0:
        return float("inf"), 0.0, df_c, df_err
    f = ms_c / ms_err
    return f, _f_pvalue(f, df_c, df_err), df_c, df_err


def _analyse_method(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute sweep signal statistics for one method."""
    by_params: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        by_params.setdefault(r["params"], []).append((r["seed"], r["val_macro_f1"]))
    configs = sorted(by_params)
    all_seeds = sorted({r["seed"] for r in rows})
    config_means: dict[str, float] = {}
    seed_groups: list[list[float]] = []
    for c in configs:
        f1s = [f for _, f in by_params[c]]
        seed_groups.append(f1s)
        if f1s:
            config_means[c] = _mean(f1s)
    if len(config_means) < 2:
        return {"n_configs": len(configs), "note": "too few configs to analyse"}
    rng = max(config_means.values()) - min(config_means.values())
    pooled_sd = _pooled_seed_sd(seed_groups)
    ratio = rng / pooled_sd if pooled_sd > 0 else float("nan")
    cell: dict[tuple[str, int], float] = {
        (c, s): f for c in configs for s, f in by_params[c]
    }
    complete = [c for c in configs if all((c, s) in cell for s in all_seeds)]
    if len(complete) >= 2 and len(all_seeds) >= 2:
        f_stat, p_val, df1, df2 = _anova_two_way(complete, all_seeds, cell)
    else:
        f_stat, p_val, df1, df2 = float("nan"), float("nan"), 0, 0
    best = max(config_means, key=lambda c: config_means[c])
    return {
        "n_configs": len(configs),
        "best_val_f1": config_means[best],
        "worst_val_f1": min(config_means.values()),
        "range": rng,
        "pooled_seed_sd": pooled_sd,
        "ratio": ratio,
        "f_stat": f_stat,
        "p_value": p_val,
        "df1": df1,
        "df2": df2,
        "optimum_at_edge": best in (configs[0], configs[-1]),
        "best_params": best,
    }


def _fmt(v: float | None, fmt: str = ".4f") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "   n/a"
    return format(v, fmt)


def _log_table(results: list[dict[str, Any]], threshold: float) -> None:
    hdr = (
        f"{'benchmark':<14} {'method':<40} {'cfgs':>5} {'range':>7} "
        f"{'seedSD':>7} {'ratio':>6} {'p':>7} {'edge':>5} {'signal':>7}"
    )
    sep = "-" * len(hdr)
    logger.info(sep)
    logger.info(hdr)
    logger.info(sep)
    for r in results:
        if "note" in r:
            logger.info(
                "%-14s %-40s  [skipped: %s]", r["benchmark"], r["method"], r["note"]
            )
            continue
        ratio = r.get("ratio", float("nan"))
        p_val = r.get("p_value", float("nan"))
        sig = "YES" if not math.isnan(ratio) and ratio >= threshold else "no"
        edge = "yes" if r.get("optimum_at_edge") else "no"
        logger.info(
            "%-14s %-40s %5d %7s %7s %6s %7s %5s %7s",
            r["benchmark"],
            r["method"],
            r["n_configs"],
            _fmt(r.get("range"), ".4f"),
            _fmt(r.get("pooled_seed_sd"), ".4f"),
            _fmt(ratio, ".2f"),
            _fmt(p_val, ".4f"),
            edge,
            sig,
        )
    logger.info(sep)
    logger.info("Signal criterion: range/seedSD >= %.1f", threshold)


def main() -> None:
    """Load tuning results from SQLite and log per-method sweep signal."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    paths = ensure_dirs(load_config(args.config))
    conn = connect(paths["db"])
    init_schema(conn)
    benchmarks = (
        ["patch_feature", "wsi_bag"] if args.benchmark == "all" else [args.benchmark]
    )
    rows = _load_tuning_f1(conn, benchmarks)
    conn.close()
    if not rows:
        logger.info("No tuning results found in database.")
        return
    by_method: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        by_method.setdefault((r["benchmark"], r["method"]), []).append(r)
    results = []
    for (benchmark, method), method_rows in sorted(by_method.items()):
        stats = _analyse_method(method_rows)
        stats.update({"benchmark": benchmark, "method": method})
        results.append(stats)
    _log_table(results, args.threshold)


if __name__ == "__main__":
    main()
