# -*- coding: utf-8 -*-
"""
Corner-case comparison figures for the SV transport poster.
Picks the 4 extreme parameter combinations (corners of the 3x3 grid)
and generates annotated figures for the Results sections.

Grid corners:
  P010  low release (0.05/s),  low ret. capture (0.25)
  P012  low release (0.05/s),  high ret. capture (0.75)
  P013  high release (0.50/s), low ret. capture (0.25)
  P015  high release (0.50/s), high ret. capture (0.75)

@author: Najmul Hasan
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import os

# ---------------------------------------------------------------------------
# Global font settings for poster readability
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family':     'Arial',
    'font.size':        13,       # base tick label size
    'axes.titlesize':   20,
    'axes.titleweight': 'bold',
    'axes.labelsize':   16,
    'xtick.labelsize':  13,
    'ytick.labelsize':  13,
    'legend.fontsize':  13,
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")

# ---------------------------------------------------------------------------
# Simulation constants (must match run_all.py / plot_poster.py)
# ---------------------------------------------------------------------------
DT = 2            # seconds per timestep
DX = 3.0          # um per presynaptic site
HOUR_STEPS = int(3600 / DT)
T_MAX = 5 * HOUR_STEPS  # 5 hours

# ---------------------------------------------------------------------------
# Corner definitions
# ---------------------------------------------------------------------------
CORNERS = [
    {"code": "P010", "rel": 0.05, "ret": 0.25,
     "label": "Low release, Low capture"},
    {"code": "P012", "rel": 0.05, "ret": 0.75,
     "label": "Low release, High capture"},
    {"code": "P013", "rel": 0.50, "ret": 0.25,
     "label": "High release, Low capture"},
    {"code": "P015", "rel": 0.50, "ret": 0.75,
     "label": "High release, High capture"},
]

# Same palette as plot_poster.py
LINE_COLORS = {
    'Moving':   '#2D6A4F',
    'Recycled': '#E76F51',
    'Pooled':   '#264653',
    'SVPs':     '#E9C46A',
}

BAR_COLORS = ['#2D6A4F', '#E76F51', '#264653', '#E9C46A']


# ---------------------------------------------------------------------------
# Data loading (same logic as plot_poster.py, copied to keep this standalone)
# ---------------------------------------------------------------------------
def load_run(sim_code):
    csv_path = os.path.join(OUTPUT_DIR, f"{sim_code}_output.csv")
    json_path = os.path.join(OUTPUT_DIR, f"{sim_code}_parameters.json")

    df = pd.read_csv(csv_path)
    with open(json_path) as f:
        params = json.load(f)

    pool_cols = [c for c in df.columns if c.startswith("PS Loc:")]
    times = np.arange(min(T_MAX, len(df)))

    pool_dists = df.loc[times, pool_cols].to_numpy()
    moving     = df.loc[times, "Axonal Transport SVs"].to_numpy()
    recycled   = df.loc[times, "Recycled SVs"].to_numpy()
    svps       = df.loc[times, "SVPs"].to_numpy()
    pooled_total = pool_dists.sum(axis=1)

    n_sites = pool_dists.shape[1]
    positions_um = np.arange(n_sites) * DX

    return {
        "pool_dists":   pool_dists,
        "moving":       moving,
        "recycled":     recycled,
        "svps":         svps,
        "pooled_total": pooled_total,
        "positions_um": positions_um,
        "params":       params,
    }


def load_corners():
    data = {}
    for c in CORNERS:
        code = c["code"]
        print(f"Loading {code} ...")
        data[code] = load_run(code)
    return data


# ---------------------------------------------------------------------------
# Figure 1: 2x2 population dynamics with annotations
# ---------------------------------------------------------------------------
def fig_population_2x2(data):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    # Layout: top row = low release, bottom row = high release
    #         left col = low capture, right col = high capture
    grid = [
        (0, 0, CORNERS[0]),   # P010 top-left
        (0, 1, CORNERS[1]),   # P012 top-right
        (1, 0, CORNERS[2]),   # P013 bottom-left
        (1, 1, CORNERS[3]),   # P015 bottom-right
    ]

    for row, col, corner in grid:
        ax = axes[row, col]
        d = data[corner["code"]]
        n_t = len(d["moving"])
        time_hrs = np.arange(n_t) * DT / 3600.0

        ax.plot(time_hrs, d["moving"],       color=LINE_COLORS['Moving'],
                lw=2.2, label='Moving')
        ax.plot(time_hrs, d["recycled"],     color=LINE_COLORS['Recycled'],
                lw=2.2, label='Recycled')
        ax.plot(time_hrs, d["pooled_total"], color=LINE_COLORS['Pooled'],
                lw=2.2, label='Pooled')
        ax.plot(time_hrs, d["svps"],         color=LINE_COLORS['SVPs'],
                lw=2.2, label='SVPs')

        ax.set_xlim(0, 5)

        # Annotation: code bold, description regular, top-right corner
        ax.text(0.97, 0.97, corner['code'],
                transform=ax.transAxes, va='top', ha='right',
                fontsize=15, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='white', edgecolor='gray', alpha=0.85))
        ax.text(0.97, 0.84, corner['label'],
                transform=ax.transAxes, va='top', ha='right',
                fontsize=13, fontweight='normal',
                bbox=dict(boxstyle='round,pad=0.2',
                          facecolor='white', edgecolor='none', alpha=0.85))

        if col == 0:
            ax.set_ylabel('Vesicle count')
        if row == 1:
            ax.set_xlabel('Time (hours)')

    # Shared legend below the title, left-aligned to avoid top-right annotations
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', ncol=2, frameon=True,
               fontsize=13, bbox_to_anchor=(0.08, 0.97))

    fig.suptitle('SV Population Dynamics',
                 fontsize=22, fontweight='bold', y=1.02)

    path = os.path.join(FIGURES_DIR, "corner_population_dynamics.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 2: Axon diagram with center-of-mass markers
# ---------------------------------------------------------------------------
def fig_axon_center_of_mass(data):
    fig, ax = plt.subplots(figsize=(13, 5))

    # Draw the axon as a thick horizontal line
    ax.plot([0, 300], [0, 0], color='#888888', lw=8, solid_capstyle='round',
            zorder=1)

    # Soma and distal markers
    ax.plot(0, 0, 's', color='#555555', markersize=16, zorder=3)
    ax.text(0, -0.50, 'Soma\n(0 um)', ha='center', va='top', fontsize=14,
            fontweight='bold')

    ax.plot(300, 0, 's', color='#555555', markersize=16, zorder=3)
    ax.text(300, -0.50, 'Distal\n(300 um)', ha='center', va='top',
            fontsize=14, fontweight='bold')

    # Midpoint reference
    ax.axvline(150, color='gray', ls='--', lw=1.2, alpha=0.6, zorder=1)
    ax.text(150, -0.50, 'Midpoint\n(150 um)', ha='center', va='top',
            fontsize=12, color='gray')

    # Compute final center of mass for each corner and place markers
    y_offsets = [0.55, 1.10, 1.65, 2.20]

    for i, corner in enumerate(CORNERS):
        d = data[corner["code"]]
        pool = d["pool_dists"]
        pos = d["positions_um"]
        pool_sum = pool[-1].sum()
        if pool_sum == 0:
            pool_sum = 1
        final_com = (pool[-1] @ pos) / pool_sum

        y = y_offsets[i]
        color = BAR_COLORS[i]

        # Vertical connector from axon to marker
        ax.plot([final_com, final_com], [0, y], color=color, lw=1.2,
                ls='--', alpha=0.6, zorder=2)

        # Marker
        ax.plot(final_com, y, 'D', color=color, markersize=12, zorder=4)

        # Label: code bold, then position and description regular
        ax.text(final_com + 5, y,
                f"{corner['code']}: {final_com:.0f} um",
                va='center', ha='left', fontsize=14, color=color,
                fontweight='bold')
        ax.text(final_com + 5, y - 0.22,
                corner['label'],
                va='center', ha='left', fontsize=12, color=color,
                fontweight='normal')

    ax.set_xlim(-15, 360)
    ax.set_ylim(-1.2, 3.0)
    ax.set_xlabel('Axon position (um)')
    ax.get_yaxis().set_visible(False)

    # Clean up spines
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)

    ax.set_title('Center of Mass Along the Axon')

    path = os.path.join(FIGURES_DIR, "axon_center_of_mass.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 3: Center-of-mass bar chart
# ---------------------------------------------------------------------------
def fig_center_of_mass_bar(data):
    fig, ax = plt.subplots(figsize=(11, 6))

    labels = []
    values = []

    for corner in CORNERS:
        d = data[corner["code"]]
        pool = d["pool_dists"]
        pos = d["positions_um"]
        pool_sum = pool[-1].sum()
        if pool_sum == 0:
            pool_sum = 1
        final_com = (pool[-1] @ pos) / pool_sum

        labels.append(f"{corner['code']}\n{corner['label'].replace(', ', '\n')}")
        values.append(final_com)

    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=BAR_COLORS, width=0.55, edgecolor='white',
                  linewidth=0.8)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                f'{val:.0f} um', ha='center', va='bottom', fontsize=15,
                fontweight='bold')

    # Midpoint reference line
    ax.axhline(150, color='gray', ls='--', lw=1.5, alpha=0.7)
    ax.text(len(labels) - 0.5, 153, 'Axon midpoint (150 um)', ha='right',
            va='bottom', fontsize=12, color='gray')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Center of mass position (um)')
    ax.set_ylim(0, 220)
    ax.set_title('Final Center of Mass at t = 5 h')

    path = os.path.join(FIGURES_DIR, "center_of_mass_bar.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    plt.style.use('seaborn-v0_8-whitegrid')
    data = load_corners()

    print("\nGenerating corner comparison figures...")
    fig_population_2x2(data)
    fig_axon_center_of_mass(data)
    fig_center_of_mass_bar(data)

    print("\nAll figures saved to figures/")
