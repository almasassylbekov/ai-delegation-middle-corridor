"""
Experiments for the Middle Corridor AI-delegation starter model.

    python run_experiments.py            # full run (~2-4 min on a laptop)
    python run_experiments.py --quick    # small grid, for testing

Outputs (in ./results):
    baseline_summary.csv / .md   architecture comparison, normal vs shift regime
    boundary_grid.csv            regret per architecture on the sigma x latency grid
    fig_delegation_boundary.png  where review (D1) beats autonomy (D2) and vice versa
    fig_regret_vs_prediction.png regret vs prediction error, per architecture
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from multiprocessing import Pool

import numpy as np
import pandas as pd

from model import Params, run_once

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ARCHS = ("D0", "D1", "D2", "D2T")


# --------------------------------------------------------------------------
def evaluate(args):
    """Regret and outcome metrics of each architecture vs ORACLE (common random numbers)."""
    p, seeds, tag = args
    rows = []
    for s in seeds:
        oracle = {r["id"]: r["cost"] for r in run_once(p, "ORACLE", s)}
        for arch in ARCHS:
            res = pd.DataFrame(run_once(p, arch, s))
            aff = res[res.affected]
            regret = (res.cost.sum() - sum(oracle[i] for i in res.id)) / max(1, len(aff))
            rv = aff[aff.reviewed]
            wrong = rv[rv.ai_wrong]
            right = rv[~rv.ai_wrong]
            rows.append({
                **tag, "arch": arch, "seed": s,
                "regret_per_affected": regret,
                "cost_per_affected": aff.cost.mean(),
                "on_time_all": res.on_time.mean(),
                "on_time_affected": aff.on_time.mean(),
                "transit_affected_h": aff.transit_h.mean(),
                "share_affected": len(aff) / len(res),
                "reroute_share": aff.rerouted.mean(),
                "reversal_share": aff.reversed.mean(),
                "reviewed_share": aff.reviewed.mean(),
                "missed_window_share": aff.missed_window.mean(),
                "catch_rate": wrong.caught.mean() if len(wrong) else np.nan,
                "false_override_rate": right.false_override.mean() if len(right) else np.nan,
            })
    return rows


def run_parallel(jobs):
    with Pool() as pool:
        out = pool.map(evaluate, jobs)
    return pd.DataFrame([r for chunk in out for r in chunk])


# --------------------------------------------------------------------------
def baseline(seeds):
    base = Params()
    jobs = [(replace(base, distribution_shift=sh), seeds,
             {"regime": "shift" if sh else "normal"}) for sh in (False, True)]
    df = run_parallel(jobs)
    cols = ["regret_per_affected", "cost_per_affected", "on_time_affected",
            "transit_affected_h", "reroute_share", "reversal_share",
            "missed_window_share", "catch_rate", "false_override_rate"]
    summary = df.groupby(["regime", "arch"])[cols].mean().round(3)
    summary.to_csv(os.path.join(OUT, "baseline_summary.csv"))
    with open(os.path.join(OUT, "baseline_summary.md"), "w") as f:
        f.write(summary.to_markdown())
    return summary


def boundary_grid(sigmas, latencies, seeds):
    base = Params()
    jobs = [(replace(base, sigma_base=s, review_mean_h=L, distribution_shift=sh), seeds,
             {"regime": "shift" if sh else "normal", "sigma": s, "review_h": L})
            for sh in (False, True) for s in sigmas for L in latencies]
    df = run_parallel(jobs)
    grid = (df.groupby(["regime", "sigma", "review_h", "arch"])["regret_per_affected"]
              .agg(["mean", "sem"]).reset_index())
    grid.to_csv(os.path.join(OUT, "boundary_grid.csv"), index=False)
    return grid


# --------------------------------------------------------------------------
# Figures (palette: validated reference instance, see README)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RED, NEUTRAL = "#e34948", "#f0efec"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def fig_boundary(grid, clip=800):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    cmap = LinearSegmentedColormap.from_list("div", [BLUE, NEUTRAL, RED])
    regimes = ["normal", "shift"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    norm = TwoSlopeNorm(vcenter=0, vmin=-clip, vmax=clip)
    for ax, r in zip(axes, regimes):
        g = grid[grid.regime == r]
        m = g.pivot_table(index="sigma", columns=["review_h", "arch"], values="mean")
        e = g.pivot_table(index="sigma", columns=["review_h", "arch"], values="sem")
        d = m.xs("D2", axis=1, level="arch") - m.xs("D1", axis=1, level="arch")
        se = np.sqrt(e.xs("D2", axis=1, level="arch") ** 2 + e.xs("D1", axis=1, level="arch") ** 2)
        im = ax.imshow(d.values.clip(-clip, clip), cmap=cmap, norm=norm,
                       origin="lower", aspect="auto")
        ax.set_xticks(range(d.shape[1]), [f"{c:g}" for c in d.columns])
        ax.set_yticks(range(d.shape[0]), [f"{i:g}" for i in d.index])
        ax.set_xlabel("Mean human review delay (hours)", color=INK2)
        ax.set_ylabel("AI prediction error (log-sd)", color=INK2)
        title = "Normal conditions" if r == "normal" else "Distribution shift (severe regime)"
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        for (i, j), v in np.ndenumerate(d.values):
            tie = abs(v) < 1.96 * se.values[i, j]
            label = "≈" if tie else ("D1" if v > 0 else "D2")
            ax.text(j, i, f"{label}\n{v:+.0f}", ha="center", va="center",
                    fontsize=7.5, color=INK, fontweight="normal" if tie else "bold")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(colors=INK2, length=0)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, extend="both")
    cb.set_label("Regret D2 − regret D1 (USD per affected movement)\n"
                 "red: human review (D1) better · blue: autonomy (D2) better", color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle("Delegation boundary for the rerouting decision "
                 "(≈ : difference within 95% CI)", color=INK, fontsize=12, x=0.01, ha="left")
    fig.savefig(os.path.join(OUT, "fig_delegation_boundary.png"), dpi=160)
    plt.close(fig)


def fig_regret_lines(grid, review_h):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, constrained_layout=True)
    series = [("D1", f"D1 AI + human review ({review_h:g} h)", BLUE, "o"),
              ("D2", "D2 autonomous execution", ORANGE, "s"),
              ("D2T", "D2T selective autonomy (confidence threshold)", AQUA, "^")]
    for ax, r in zip(axes, ["normal", "shift"]):
        g = grid[(grid.regime == r) & (grid.review_h == review_h)]
        for arch, label, color, marker in series:
            d = g[g.arch == arch].sort_values("sigma")
            ax.plot(d.sigma, d["mean"], color=color, lw=2, marker=marker, ms=7,
                    label=label)
            ax.fill_between(d.sigma, d["mean"] - 1.96 * d["sem"], d["mean"] + 1.96 * d["sem"],
                            color=color, alpha=0.12, lw=0)
        ax.set_title("Normal conditions" if r == "normal" else "Distribution shift",
                     color=INK, fontsize=11, loc="left")
        ax.set_xlabel("AI prediction error (log-sd)", color=INK2)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(INK2)
        ax.tick_params(colors=INK2, length=0)
    axes[0].set_ylabel("Decision regret vs oracle\n(USD per affected movement)", color=INK2)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK)
    fig.suptitle("Regret grows with prediction error — faster for autonomy than for review",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.savefig(os.path.join(OUT, "fig_regret_vs_prediction.png"), dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small grid for testing")
    ap.add_argument("--seeds", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.quick:
        seeds = range(a.seeds or 3)
        sigmas, latencies = [0.2, 0.75, 1.5], [2, 12, 48]
    else:
        seeds = range(a.seeds or 10)
        sigmas = [0.2, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5]
        latencies = [1, 3, 6, 12, 24, 48]

    print("Baseline comparison ...")
    print(baseline(seeds).to_string())
    print("\nDelegation-boundary grid ...")
    grid = boundary_grid(sigmas, latencies, seeds)
    plot(grid, latencies)
    print(f"\nDone. Results in {OUT}")


def plot(grid, latencies):
    fig_boundary(grid)
    fig_regret_lines(grid, review_h=6 if 6 in latencies else latencies[0])


if __name__ == "__main__":
    main()
