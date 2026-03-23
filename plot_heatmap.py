# -*- coding: utf-8 -*-
"""
Single-run spatiotemporal heatmap of pooled SV distribution.
Reads baseline_output.csv from the script directory.

@author: Najmul Hasan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(SCRIPT_DIR, "baseline_output.csv")

df = pd.read_csv(csv_path)

pool_cols = [c for c in df.columns if c.startswith("PS Loc:")]

# Pool distributions over 5 hours
hour = 60 * 60 / 2  # timesteps per hour
times = np.arange(0, 5 * hour - 1)
pool_dists = df.loc[times, pool_cols].to_numpy()

dt = 2.0  # seconds per timestep
n_times, n_positions = pool_dists.shape

y_max = 5  # hours
dx = 3.0   # um per presynapse
x_min = 0
x_max = (n_positions - 1) * dx

plt.figure(figsize=(10, 5))
plt.imshow(pool_dists, aspect="auto", origin="lower",
           extent=[x_min, x_max, 0, y_max],
           interpolation="nearest")
plt.yticks(range(0, 6))
plt.colorbar(label="SV count")
plt.xlabel("Distance along axon (um)")
plt.ylabel("Time (hours)")
plt.tight_layout()
plt.show()
