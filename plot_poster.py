# -*- coding: utf-8 -*-
"""
Poster-quality combined figures for the SV transport parameter sweep.
Reads all 9 simulation outputs and generates 3x3 comparison panels.

Grid layout:
  Columns: p_cap_fast_ret = 0.25, 0.50, 0.75  (retrograde capture probability)
  Rows:    p_rel_multiplier = 0.05, 0.20, 0.50 (release rate /s)

@author: Najmul Hasan
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Parameter grid: rows = release rate, columns = retrograde capture
RELEASE_RATES = [0.05, 0.20, 0.50]
RET_CAPTURES = [0.25, 0.50, 0.75]

# Sim code mapping (from run_all.py RUNS list)
CODE_GRID = {
    (0.05, 0.25): "P010", (0.05, 0.50): "P011", (0.05, 0.75): "P012",
    (0.20, 0.25): "P001", (0.20, 0.50): "P002", (0.20, 0.75): "P003",
    (0.50, 0.25): "P013", (0.50, 0.50): "P014", (0.50, 0.75): "P015",
}

# Simulation constants
DT = 2          # seconds per timestep
DX = 3.0        # um per presynaptic site
HOUR_STEPS = int(3600 / DT)  # timesteps per hour
T_MAX = 5 * HOUR_STEPS       # 5 hours of data

# Row / column labels for the grid
ROW_LABELS = [f"Release = {r}/s" for r in RELEASE_RATES]
COL_LABELS = [f"Ret. capture = {c}" for c in RET_CAPTURES]


def load_run(sim_code):
    """Load CSV and parameter JSON for one run."""
    csv_path = os.path.join(OUTPUT_DIR, f"{sim_code}_output.csv")
    json_path = os.path.join(OUTPUT_DIR, f"{sim_code}_parameters.json")

    df = pd.read_csv(csv_path)
    with open(json_path) as f:
        params = json.load(f)

    pool_cols = [c for c in df.columns if c.startswith("PS Loc:")]
    times = np.arange(min(T_MAX, len(df)))

    pool_dists = df.loc[times, pool_cols].to_numpy()
    moving = df.loc[times, "Axonal Transport SVs"].to_numpy()
    recycled = df.loc[times, "Recycled SVs"].to_numpy()
    svps = df.loc[times, "SVPs"].to_numpy()
    pooled_total = pool_dists.sum(axis=1)

    n_sites = pool_dists.shape[1]
    positions_um = np.arange(n_sites) * DX

    return {
        "pool_dists": pool_dists,
        "moving": moving,
        "recycled": recycled,
        "svps": svps,
        "pooled_total": pooled_total,
        "positions_um": positions_um,
        "params": params,
    }


def load_all():
    """Load all 9 runs into a dict keyed by (rel, ret)."""
    data = {}
    for (rel, ret), code in CODE_GRID.items():
        print(f"Loading {code} (rel={rel}, ret={ret})...")
        data[(rel, ret)] = load_run(code)
    return data


# ---------------------------------------------------------------------------
# Figure 1: 3x3 spatiotemporal heatmaps
# ---------------------------------------------------------------------------
def fig_heatmap_grid(data):
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), constrained_layout=True)

    # Find global vmin/vmax for consistent colorbar
    vmax = max(d["pool_dists"].max() for d in data.values())

    for row_i, rel in enumerate(RELEASE_RATES):
        for col_i, ret in enumerate(RET_CAPTURES):
            ax = axes[row_i, col_i]
            d = data[(rel, ret)]
            pool = d["pool_dists"]
            n_sites = pool.shape[1]
            x_max = (n_sites - 1) * DX

            im = ax.imshow(pool, aspect='auto', origin='lower',
                           extent=[0, x_max, 0, 5],
                           interpolation='bilinear', cmap='inferno',
                           vmin=0, vmax=vmax)
            ax.set_yticks(range(6))

            code = CODE_GRID[(rel, ret)]
            ax.set_title(code, fontsize=12, fontweight='bold')

            if col_i == 0:
                ax.set_ylabel(f'{ROW_LABELS[row_i]}\nTime (hours)', fontsize=11)
            else:
                ax.set_yticklabels([])

            if row_i == 2:
                ax.set_xlabel('Axon position (µm)', fontsize=11)
            else:
                ax.set_xticklabels([])

    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    cb.set_label('Vesicle count', fontsize=12)

    fig.suptitle('Pooled SV Distribution, Parameter Sweep',
                 fontsize=16, fontweight='bold', y=1.01)

    path = os.path.join(FIGURES_DIR, "poster_heatmap_3x3.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 2: 3x3 final distribution stem plots
# ---------------------------------------------------------------------------
def fig_final_dist_grid(data):
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), constrained_layout=True)

    ymax = max(d["pool_dists"][-1].max() for d in data.values()) * 1.1

    for row_i, rel in enumerate(RELEASE_RATES):
        for col_i, ret in enumerate(RET_CAPTURES):
            ax = axes[row_i, col_i]
            d = data[(rel, ret)]
            final = d["pool_dists"][-1]
            pos = d["positions_um"]

            markerline, stemlines, baseline = ax.stem(
                pos, final, linefmt='#264653', markerfmt='o', basefmt='gray')
            plt.setp(stemlines, linewidth=0.6, alpha=0.7)
            plt.setp(markerline, markersize=2, color='#2D6A4F')

            ax.set_ylim(0, ymax)
            code = CODE_GRID[(rel, ret)]
            ax.set_title(code, fontsize=12, fontweight='bold')

            if col_i == 0:
                ax.set_ylabel(f'{ROW_LABELS[row_i]}\nVesicle count', fontsize=11)
            else:
                ax.set_yticklabels([])

            if row_i == 2:
                ax.set_xlabel('Axon position (µm)', fontsize=11)
            else:
                ax.set_xticklabels([])

    fig.suptitle('Final SV Distribution at t = 5 h, Parameter Sweep',
                 fontsize=16, fontweight='bold', y=1.01)

    path = os.path.join(FIGURES_DIR, "poster_final_dist_3x3.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 3: 3x3 center-of-mass trajectories
# ---------------------------------------------------------------------------
def fig_center_mass_grid(data):
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), constrained_layout=True)

    for row_i, rel in enumerate(RELEASE_RATES):
        for col_i, ret in enumerate(RET_CAPTURES):
            ax = axes[row_i, col_i]
            d = data[(rel, ret)]
            pool = d["pool_dists"]
            pos = d["positions_um"]
            n_t = pool.shape[0]
            time_hrs = np.arange(n_t) * DT / 3600.0

            pool_sums = pool.sum(axis=1)
            pool_sums[pool_sums == 0] = 1
            mean_pos = (pool @ pos) / pool_sums

            ax.plot(time_hrs, mean_pos, color='#E76F51', linewidth=1.5)
            ax.axhline(150, color='gray', ls='--', lw=0.8, alpha=0.6)
            ax.set_ylim(0, 300)
            ax.set_xlim(0, 5)

            code = CODE_GRID[(rel, ret)]
            ax.set_title(code, fontsize=12, fontweight='bold')

            if col_i == 0:
                ax.set_ylabel(f'{ROW_LABELS[row_i]}\nMean position (µm)', fontsize=11)
            else:
                ax.set_yticklabels([])

            if row_i == 2:
                ax.set_xlabel('Time (hours)', fontsize=11)
            else:
                ax.set_xticklabels([])

    fig.suptitle('Center of Mass, Parameter Sweep',
                 fontsize=16, fontweight='bold', y=1.01)

    path = os.path.join(FIGURES_DIR, "poster_center_mass_3x3.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 4: 3x3 population dynamics
# ---------------------------------------------------------------------------
def fig_populations_grid(data):
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), constrained_layout=True)
    COLORS = {'Moving': '#2D6A4F', 'Recycled': '#E76F51',
              'Pooled': '#264653', 'SVPs': '#E9C46A'}

    for row_i, rel in enumerate(RELEASE_RATES):
        for col_i, ret in enumerate(RET_CAPTURES):
            ax = axes[row_i, col_i]
            d = data[(rel, ret)]
            n_t = len(d["moving"])
            time_hrs = np.arange(n_t) * DT / 3600.0

            ax.plot(time_hrs, d["moving"], color=COLORS['Moving'],
                    lw=1.2, label='Moving')
            ax.plot(time_hrs, d["recycled"], color=COLORS['Recycled'],
                    lw=1.2, label='Recycled')
            ax.plot(time_hrs, d["pooled_total"], color=COLORS['Pooled'],
                    lw=1.2, label='Pooled')
            ax.plot(time_hrs, d["svps"], color=COLORS['SVPs'],
                    lw=1.2, label='SVPs')

            ax.set_xlim(0, 5)
            code = CODE_GRID[(rel, ret)]
            ax.set_title(code, fontsize=12, fontweight='bold')

            if col_i == 0:
                ax.set_ylabel(f'{ROW_LABELS[row_i]}\nVesicle count', fontsize=11)
            else:
                ax.set_yticklabels([])

            if row_i == 2:
                ax.set_xlabel('Time (hours)', fontsize=11)
            else:
                ax.set_xticklabels([])

            if row_i == 0 and col_i == 2:
                ax.legend(fontsize=8, loc='upper right', frameon=True)

    fig.suptitle('Population Dynamics, Parameter Sweep',
                 fontsize=16, fontweight='bold', y=1.01)

    path = os.path.join(FIGURES_DIR, "poster_populations_3x3.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    plt.style.use('seaborn-v0_8-whitegrid')
    data = load_all()

    print("\nGenerating poster figures...")
    fig_heatmap_grid(data)
    fig_final_dist_grid(data)
    fig_center_mass_grid(data)
    fig_populations_grid(data)

    print("\nAll poster figures saved to figures/")
