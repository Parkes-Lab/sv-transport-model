# -*- coding: utf-8 -*-
"""
Synaptic vesicle (SV) population transport simulation.
Tracks SV movement along an axon with presynaptic capture/release dynamics.

@author: Mason Parkes, Najmul Hasan
"""

import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import json
import os

# Output config - uses script directory by default
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_NAME = "baseline"  # change this for each run (e.g., "low_ret_capture", "fast_release")
CSV_NAME = f"{RUN_NAME}_output.csv"
PARAM_NAME = f"{RUN_NAME}_parameters.json"

PRINT_DEBUG = False
PLOT_EACH_STEP = False

# Simulation parameters
params = {
    "version": "SynPopSim_MP_v1",
    "axon_length": 300,          # um total length
    "ps_spacing": 3,             # presynaptic site every 3 um
    "num_slow_SVs": 10,
    "num_fast_SVs": 10,
    "time_step_size": 2,         # seconds per step
    "time_steps": 9000,
    "time_interval": 0.3,        # animation pause (sec)
    "StartingPoolSize": 40,
    "p_ant": 0.50,               # probability anterograde vs retrograde
    "p_slow": 0.8,               # probability of slow transport on release
    "p_cap_slow": 0.75,          # capture prob: slow SVs
    "p_cap_fast_ant": 0.75,      # capture prob: fast anterograde
    "p_cap_fast_ret": 0.75,      # capture prob: fast retrograde
    "svp_cap": 0.01,             # SVP capture prob (low to ensure delivery through axon)
    "faststep_multiplier": 1.5,  # um/s fast (MT-based)
    "slowstep_multiplier": 0.5,  # um/s slow (actin-based)
    "p_rel_multiplier": 0.2,     # release prob per second
    "rate_bg_multiplier": 4,     # SVPs created per second at soma (range: 4-15/s, Parkes et al. 2023)
    "notes": "Single-run baseline configuration"
}

# Derived quantities (scaled by time step)
dt = params["time_step_size"]
faststep = params["faststep_multiplier"] * dt
slowstep = params["slowstep_multiplier"] * dt
p_rel = params["p_rel_multiplier"] * dt
rate_bg = int(params["rate_bg_multiplier"] * dt)

# Presynaptic site grid: 0, 3, 6, ..., 300
ps_positions = np.arange(0, params["axon_length"] + params["ps_spacing"], params["ps_spacing"])
num_sites = len(ps_positions)
ps_set = set(ps_positions)

# Tracking arrays
n_steps = params["time_steps"]
pools = np.zeros([num_sites, n_steps], dtype=int)
moving = np.zeros(n_steps)
recycled = np.zeros_like(moving)
totals = np.zeros_like(moving)
SVPs = np.zeros_like(moving)


def is_presyn(x):
    return x in ps_set


def random_direction():
    return 1 if rnd.random() > params["p_ant"] else -1


def make_sv(pos, sv_type, direction, state='moving'):
    return {'pos': float(pos), 'type': sv_type, 'dir': direction, 'state': state}


# Delivery metrics
arrivals_presyn_total = 0
captured_presyn_total = 0

# Initial SV population
SVs = []

for _ in range(params["num_slow_SVs"]):
    SVs.append(make_sv(rnd.randint(0, params["axon_length"]), 'slow', random_direction()))

for _ in range(params["num_fast_SVs"]):
    SVs.append(make_sv(rnd.randint(0, params["axon_length"]), 'fast', random_direction()))

for pos in ps_positions:
    for _ in range(params["StartingPoolSize"]):
        SVs.append(make_sv(pos, 'fast', 1, state='pooled'))

# Plotting setup
plt.ion()
FIGSIZE = (10, 3.0)
axon_len = params["axon_length"]
spacing = params["ps_spacing"]

# Main loop
for t in range(n_steps):
    pct = 1 + 100 * t // n_steps
    print(f"{pct:03d}% complete", end='\r', flush=True)

    # Create new SVPs at soma
    for _ in range(rate_bg):
        SVs.append(make_sv(0.0, 'fast', 1, state='SVP'))

    # Record current state
    for sv in SVs:
        if sv['state'] == 'pooled':
            pools[int(sv['pos'] / spacing), t] += 1
        elif sv['state'] == 'moving':
            moving[t] += 1
        elif sv['state'] == 'recycled':
            recycled[t] += 1
        elif sv['state'] == 'SVP':
            SVPs[t] += 1
    totals[t] = int(np.sum(pools[:, t]) + moving[t] + recycled[t])

    if PRINT_DEBUG:
        print(
            f"Time step {t:03d}, SV# {len(SVs):04d} "
            f"Pooled: {int(np.sum(pools[:, t])):04d} "
            f"Moving: {int(moving[t]):04d} "
            f"Recycled: {int(recycled[t]):04d} "
            f"Total: {int(totals[t]):04d}"
        )

    # Process each SV
    for sv in SVs:
        if sv['state'] == 'moving':
            step = faststep if sv['type'] == 'fast' else slowstep
            newpos = sv['pos'] + step * sv['dir']

            if newpos > axon_len:
                # Reflect at distal end
                newpos = axon_len - (newpos - axon_len)
                sv['dir'] = -1
            elif newpos < 0:
                # Recycle at soma
                sv['pos'] = 0
                sv['state'] = 'recycled'
                continue
            else:
                sv['pos'] = newpos

            # Try capture at presynaptic sites
            if is_presyn(sv['pos']):
                arrivals_presyn_total += 1

                if sv['type'] == 'slow':
                    p_cap = params["p_cap_slow"]
                elif sv['dir'] == 1:
                    p_cap = params["p_cap_fast_ant"]
                else:
                    p_cap = params["p_cap_fast_ret"]

                if rnd.random() < p_cap:
                    sv['state'] = 'pooled'
                    captured_presyn_total += 1

        elif sv['state'] == 'pooled':
            if rnd.random() < p_rel:
                sv['state'] = 'moving'
                sv['type'] = 'slow' if rnd.random() < params["p_slow"] else 'fast'
                sv['dir'] = random_direction()

        elif sv['state'] == 'SVP':
            newpos = sv['pos'] + faststep * sv['dir']
            if newpos > axon_len:
                newpos = axon_len - (newpos - axon_len)
                sv['dir'] = -1
            else:
                sv['pos'] = newpos
            if is_presyn(newpos):
                if rnd.random() < params["svp_cap"]:
                    sv['state'] = 'pooled'

    if PLOT_EACH_STEP:
        plt.cla()
        plt.plot(pools[:, t])
        plt.ylim((0, 80))
        plt.pause(.1)

print("\n100% complete")

# Export results
export = np.row_stack((totals, SVPs, moving, recycled, pools))
header = "Total SVs,SVPs,Axonal Transport SVs,Recycled SVs," + ",".join(f"PS Loc: {ps}" for ps in ps_positions)

csv_path = os.path.join(SCRIPT_DIR, CSV_NAME)
params_path = os.path.join(SCRIPT_DIR, PARAM_NAME)

np.savetxt(csv_path, export.T, delimiter=",", fmt="%d", header=header, comments="")
with open(params_path, "w") as f:
    json.dump(params, f, indent=2)

print(f"Output saved: {csv_path}")
print(f"Params saved: {params_path}")

# Final plot
plt.ioff()
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.plot(moving)
ax.set_ylim((0, max(80, int(moving.max()) + 5)))
ax.set_xlabel('Time step')
ax.set_ylabel('Moving SVs')
ax.set_title('Moving SVs over time')
plt.show()
