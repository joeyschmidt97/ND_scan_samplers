"""Figures for the compute-sink and wrong-mode sweeps.

Reads the `stats-end_to_end_error.json` summaries written by `sink_stats` and
draws three figures into `--output`:

    sink-error.png        geometric-mean end-to-end error vs c_max, 2D and 6D
    sink-share.png        share of compute spent on contested points vs c_max
    wrongmode-error.png   6D end-to-end error vs flip_peak at c_max 1 and 4,
                          flip_peak 0 taken from the 6D sink sweep

    python -m ND_scan_samplers.scripts.analysis.plot_sink --output ND_scan_samplers/results/wrongmode-2026-11-02/figures
"""
import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ND_scan_samplers import ROOT

# Fixed categorical order (validated light palette); markers carry identity too.
ARMS = (("vurs", "#2a78d6", "o"), ("space-filling", "#eb6834", "s"), ("vurs-cost", "#1baf7a", "^"),
        ("vurs-prod", "#eda100", "D"), ("vurs-flag", "#e87ba4", "v"), ("vurs-avoid", "#008300", "P"),
        ("vurs-a", "#6250d6", "X"))
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3de"
SINK = ROOT/"results/sink-2026-11-01"
WRONG = ROOT/"results/wrongmode-2026-11-02"


def summary(path):
    return json.loads((path/"stats-end_to_end_error.json").read_text(encoding="utf-8"))["summary"]


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=.8)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def lines(ax, xs, values):
    for arm, color, marker in ARMS:
        ax.plot(xs, values(arm), color=color, marker=marker, markersize=6, linewidth=2,
                markeredgecolor="white", markeredgewidth=1, label=arm)


def sink_figure(field, ylabel, name, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="white")
    costs = (1, 2, 4, 8)
    for ax, dim in zip(axes, ("2d", "6d")):
        table = summary(SINK/dim)
        lines(ax, costs, lambda arm: [table[f"{arm}|{c:g}"][field] for c in costs])
        ax.set_xscale("log", base=2)
        ax.set_xticks(costs, [f"{c}x" for c in costs])
        style(ax, f"{dim.upper()}", "contested-run cost $c_{max}$", ylabel)
    axes[1].legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(output/name, dpi=160)
    plt.close(fig)


def wrongmode_figure(output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="white", sharey=True)
    control, table = summary(SINK/"6d"), summary(WRONG)
    flips = (0., .25, .5)
    for ax, c in zip(axes, (1, 4)):
        lines(ax, flips, lambda arm: [control[f"{arm}|{c:g}"]["error"]] +
              [table[f"{arm}|p={p:g},c={c:g}"]["error"] for p in flips[1:]])
        ax.set_xticks(flips, ["0 (sink sweep)", "0.25", "0.5"])
        style(ax, f"6D, $c_{{max}}$ = {c}x", "wrong-branch probability at a tie (flip_peak)",
              "end-to-end error (geometric mean)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(output/"wrongmode-error.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sink_figure("error", "end-to-end error (geometric mean)", "sink-error.png", args.output)
    sink_figure("sink_share", "share of compute on contested points", "sink-share.png", args.output)
    wrongmode_figure(args.output)


if __name__ == "__main__":
    main()
