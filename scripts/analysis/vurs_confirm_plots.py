"""Figures for the VURS confirmation (dev pilot or confirmation run).

Reads the outputs of `vurs_confirm_stats` (board.json, power.json) in a results
directory and writes four PNGs beside them:

    geometries.png   four example cases per stratum (gamma, dominant-mode switch)
    board.png        C1 difference and C2 geometric-mean ratio per comparator, 95% bootstrap-t
    width.png        C2-w: block error ratio against peak width l, per comparator
    power.png        C2 power at the planning alternative against blocks per stratum

    python -m ND_scan_samplers.scripts.analysis.vurs_confirm_plots --results results/vurs-confirm-dev-2026-10-25
"""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ND_scan_samplers.scripts.analysis.vurs_confirm_stats import COMPARATORS, MARGIN
from ND_scan_samplers.scripts.data.confirmation_cases import (BLOCKS_PER_STRATUM, DEV_ENTROPY, STRATA,
                                                              ConfirmationCase)

# validated categorical slots 1-4 (dataviz reference palette, light mode); a
# marker shape per stratum keeps identity off colour alone
STRATUM_STYLE = dict(smooth=("#2a78d6", "o"), curved=("#eb6834", "s"),
                     narrow=("#1baf7a", "^"), islands=("#eda100", "D"))
COMPARATOR_COLOR = dict(zip(COMPARATORS, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")))
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


def _style(ax):
    ax.grid(color=GRID, lw=.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def geometries(path, entropy=DEV_ENTROPY, per_stratum=4):
    n = 200
    grid = np.linspace(0, 1, n)
    x = np.array(np.meshgrid(grid, grid)).reshape(2, -1).T
    fig, axes = plt.subplots(len(STRATA), per_stratum, figsize=(2.6*per_stratum, 2.7*len(STRATA)))
    for i, stratum in enumerate(STRATA):
        for j in range(per_stratum):
            case = ConfirmationCase(entropy, BLOCKS_PER_STRATUM*i+j)
            ax = axes[i, j]
            ax.imshow(case(x).reshape(n, n), origin="lower", extent=(0, 1, 0, 1), cmap="viridis")
            ax.contour(grid, grid, case.region(x).reshape(n, n), levels=np.arange(case.n_modes)+.5,
                       colors="white", linewidths=1.2)
            ax.set_title(f"{stratum}, block {case.block}\nK = {case.n_modes}, l = {case.peak_width():.3f}",
                         fontsize=8, color=INK)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Development geometries: gamma (colour), dominant-mode switch (white)", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def board(rows, path, label):
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6))
    y = np.arange(len(COMPARATORS))[::-1]
    for ax, endpoint in ((left, "c1"), (right, "c2")):
        part = {r["comparator"]: r for r in rows if r["endpoint"] == endpoint}
        key = ("estimate", "low", "high") if endpoint == "c1" else ("ratio", "ratio_low", "ratio_high")
        for yi, c in zip(y, COMPARATORS):
            r = part[c]
            ax.plot([r[key[1]], r[key[2]]], [yi, yi], color=INK, lw=2, solid_capstyle="round")
            ax.plot(r[key[0]], yi, "o", ms=8, color="#2a78d6" if r["superior"] else "white",
                    mec=INK, mew=1.5)
        null = 0 if endpoint == "c1" else 1
        margin = MARGIN["c1"] if endpoint == "c1" else float(np.exp(MARGIN["c2"]))
        ax.axvline(null, color=MUTED, lw=1)
        ax.axvline(margin, color=INK, lw=1, ls="--")
        ax.set_yticks(y, COMPARATORS)
        _style(ax)
        if endpoint == "c2":
            ax.set_xscale("log")
            ticks = [.2, .3, .5, .7, .9, 1., 1.5]
            ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
            ax.minorticks_off()
            ax.set_xlabel("C2: geometric-mean error ratio VURS / comparator (N = 256)", fontsize=9, color=INK)
            ax.set_yticklabels([])
        else:
            ax.set_xlabel("C1: label surprise VURS - comparator, bits (N = 32, 64, 98)", fontsize=9, color=INK)
        ax.text(margin, len(COMPARATORS)-.4, " margin", fontsize=8, color=INK, va="bottom")
    fig.suptitle(f"{label}: VURS vs each comparator, 95% bootstrap-t (filled = practical superiority "
                 f"after Holm)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def width(detail, slopes, path, label):
    slopes = {s["comparator"]: s for s in slopes}
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), sharex=True, sharey=True)
    lx = np.log([d["peak_width"] for d in detail])
    for ax, c in zip(axes.flat, COMPARATORS):
        ly = np.array([d["c2_log_ratio"][c] for d in detail])
        for stratum in STRATA:
            rows = [i for i, d in enumerate(detail) if d["stratum"] == stratum]
            color, marker = STRATUM_STYLE[stratum]
            ax.scatter(np.exp(lx[rows]), np.exp(ly[rows]), s=40, color=color, marker=marker,
                       edgecolor="white", linewidth=1, label=stratum, zorder=3)
        fit = np.polyfit(lx, ly, 1)
        xs = np.linspace(lx.min(), lx.max(), 50)
        ax.plot(np.exp(xs), np.exp(np.polyval(fit, xs)), color=INK, lw=2)
        ax.axhline(1, color=MUTED, lw=1)
        ax.axhline(.9, color=INK, lw=1, ls="--")
        s = slopes[c]
        ax.set_title(f"vs {c}\nslope {s['slope']:.2f} [{s['low']:.2f}, {s['high']:.2f}]", fontsize=9, color=INK)
        ax.set_xscale("log")
        ax.set_yscale("log")
        _style(ax)
    for ax in axes[1]:
        ax.set_xlabel("peak width l (fit-free, mass-weighted)", fontsize=9, color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("error ratio VURS / comparator", fontsize=9, color=INK)
    axes[0, 0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle(f"{label}: C2-w, VURS's peak-error advantage against peak width (below 1 = VURS better)",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def power(rows, path, label):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ends = {}
    for c in COMPARATORS:
        for inflate, ls, lw in ((1., "-", 2), (1.25, ":", 1.5)):
            part = sorted((r["per_stratum"], r["reject_bonferroni"]) for r in rows
                          if r["endpoint"] == "c2" and r["comparator"] == c and r["inflate"] == inflate)
            n, p = zip(*part)
            ax.plot(n, p, ls=ls, lw=lw, color=COMPARATOR_COLOR[c], marker="o" if inflate == 1 else None,
                    ms=8)
        ends[c] = [r["reject_bonferroni"] for r in rows if r["endpoint"] == "c2" and r["comparator"] == c
                   and r["inflate"] == 1. and r["per_stratum"] == n[-1]][0]
    # direct labels, pushed apart where line ends collide
    placed = []
    for c in sorted(ends, key=ends.get):
        y = max([ends[c]]+[q+.045 for q in placed])
        placed.append(y)
        ax.text(n[-1]+.6, y, f"{c} {ends[c]:.2f}", fontsize=8, color=INK, va="center")
    ax.axhline(.8, color=INK, lw=1, ls="--")
    ax.text(18.3, .815, "80% target", fontsize=8, color=INK)
    ax.set_xticks([20, 30, 40])
    ax.set_xlim(18, 50)
    ax.set_ylim(0, 1)
    ax.set_xlabel("independent blocks per stratum (x4 strata)", fontsize=9, color=INK)
    ax.set_ylabel("power, Holm first step (alpha/12)", fontsize=9, color=INK)
    _style(ax)
    fig.suptitle(f"{label}: C2 power at R = 0.80; solid = dev spread, dotted = spread x1.25\n"
                 f"(C1 power is 1.00 at every size)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    root = pathlib.Path(args.results)
    board_data = json.loads((root/"board.json").read_text())
    label = "DEV pilot" if board_data["namespace"] == "dev" else "Confirmation"
    if board_data["namespace"] == "dev":
        geometries(root/"geometries.png")
    board(board_data["rows"], root/"board.png", label)
    width(board_data["blocks_detail"], board_data["width_slopes"], root/"width.png", label)
    if (root/"power.json").exists():
        power(json.loads((root/"power.json").read_text())["rows"], root/"power.png", label)


if __name__ == "__main__":
    main()
