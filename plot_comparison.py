# -*- coding: utf-8 -*-
"""
Side-by-side heatmap comparison of two simulation runs.
Reads baseline and low_ret_capture CSVs from the script directory.

@author: Najmul Hasan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to the two runs being compared
csv_path_aa = os.path.join(SCRIPT_DIR, "baseline_output.csv")
csv_path_ad = os.path.join(SCRIPT_DIR, "low_ret_capture_output.csv")

df_aa = pd.read_csv(csv_path_aa)
df_ad = pd.read_csv(csv_path_ad)

# Pool columns
pool_cols_aa = [c for c in df_aa.columns if c.startswith("PS Loc:")]
pool_cols_ad = [c for c in df_ad.columns if c.startswith("PS Loc:")]

if pool_cols_aa != pool_cols_ad:
    raise ValueError("AA and AD pool columns do not match.")

pool_cols = pool_cols_aa

# Time handling
hour = 60 * 60 / 2   # timesteps per hour
times = np.arange(0, 5 * hour - 1)

pool_dists_aa = df_aa.loc[times, pool_cols].to_numpy()
pool_dists_ad = df_ad.loc[times, pool_cols].to_numpy()

# Axes scaling
dx = 3.0  # um per presynapse
x_min = 0
x_max = (pool_dists_aa.shape[1] - 1) * dx
y_max = 5  # hours

# Plot colormaps
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

im0 = axes[0].imshow(
    pool_dists_aa,
    aspect="auto",
    origin="lower",
    extent=[x_min, x_max, 0, y_max],
    interpolation="nearest"
)
axes[0].set_xlabel("Distance along axon (um)")
axes[0].set_ylabel("Time (hours)")
axes[0].set_yticks(range(0, 6))
fig.colorbar(im0, ax=axes[0], shrink=0.9, label="SV count")

im1 = axes[1].imshow(
    pool_dists_ad,
    aspect="auto",
    origin="lower",
    extent=[x_min, x_max, 0, y_max],
    interpolation="nearest"
)
axes[1].set_xlabel("Distance along axon (um)")
axes[1].set_ylabel("Time (hours)")
axes[1].set_yticks(range(0, 6))
fig.colorbar(im1, ax=axes[1], shrink=0.9, label="SV count")

plt.tight_layout()
plt.show()
