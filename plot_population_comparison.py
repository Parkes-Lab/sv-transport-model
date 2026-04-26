# -*- coding: utf-8 -*-
"""
Compare one SV population type across the 4 corner conditions.
Generates 3 figures: Moving, Recycled, and Pooled SVs over time,
each with 4 lines (P010, P012, P013, P015).

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
import json
import os

# ---------------------------------------------------------------------------
# Global font settings (same as plot_corner_comparison.py)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family':     'Arial',
    'font.size':        13,
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
# Simulation constants
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

# One color per condition
CONDITION_COLORS = ['#2D6A4F', '#E76F51', '#264653', '#E9C46A']


# ---------------------------------------------------------------------------
# Data loading
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

    return {
        "moving":       moving,
        "recycled":     recycled,
        "svps":         svps,
        "pooled_total": pooled_total,
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
# Single population comparison plot
# ---------------------------------------------------------------------------
def fig_population_comparison(data, population_key, title, filename):
    """Plot one population type across all 4 corner conditions."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, corner in enumerate(CORNERS):
        d = data[corner["code"]]
        n_t = len(d[population_key])
        time_hrs = np.arange(n_t) * DT / 3600.0

        ax.plot(time_hrs, d[population_key],
                color=CONDITION_COLORS[i], lw=2.2,
                label=f"{corner['code']} ({corner['label']})")

    ax.set_xlim(0, 5)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Vesicle count')
    ax.set_title(title)
    ax.legend(loc='best', frameon=True)

    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    plt.style.use('seaborn-v0_8-whitegrid')
    data = load_corners()

    print("\nGenerating population comparison figures...")

    fig_population_comparison(
        data, "moving",
        "Moving SVs Over Time",
        "moving_sv_comparison.png")

    fig_population_comparison(
        data, "recycled",
        "Recycled SVs Over Time",
        "recycled_sv_comparison.png")

    fig_population_comparison(
        data, "pooled_total",
        "Pooled SVs Over Time",
        "pooled_sv_comparison.png")

    print("\nAll population comparison figures saved to figures/")
