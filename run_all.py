# -*- coding: utf-8 -*-
"""
Batch runner for all 9 assigned parameter sets.
Runs each simulation and generates figures, saving everything to output/ and figures/.

@author: Najmul Hasan
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random as rnd
import json
import os
from datetime import date

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# All 9 assigned parameter sets (from professor's spreadsheet)
# Columns: Ret_Fast_Capture, P_Fast, SV_release_rate, Sim_Code
RUNS = [
    {"p_cap_fast_ret": 0.25, "p_slow": 0.8, "p_rel_multiplier": 0.05, "sim_code": "P010"},
    {"p_cap_fast_ret": 0.50, "p_slow": 0.8, "p_rel_multiplier": 0.05, "sim_code": "P011"},
    {"p_cap_fast_ret": 0.75, "p_slow": 0.8, "p_rel_multiplier": 0.05, "sim_code": "P012"},
    {"p_cap_fast_ret": 0.25, "p_slow": 0.8, "p_rel_multiplier": 0.50, "sim_code": "P013"},
    {"p_cap_fast_ret": 0.50, "p_slow": 0.8, "p_rel_multiplier": 0.50, "sim_code": "P014"},
    {"p_cap_fast_ret": 0.75, "p_slow": 0.8, "p_rel_multiplier": 0.50, "sim_code": "P015"},
    {"p_cap_fast_ret": 0.25, "p_slow": 0.8, "p_rel_multiplier": 0.20, "sim_code": "P001"},
    {"p_cap_fast_ret": 0.50, "p_slow": 0.8, "p_rel_multiplier": 0.20, "sim_code": "P002"},
    {"p_cap_fast_ret": 0.75, "p_slow": 0.8, "p_rel_multiplier": 0.20, "sim_code": "P003"},
]


def run_simulation(p_cap_fast_ret, p_slow, p_rel_multiplier, sim_code):
    """Run one simulation with the given parameters and save output."""
    print(f"\n{'='*60}")
    print(f"Starting run {sim_code}: ret_cap={p_cap_fast_ret}, p_slow={p_slow}, rel={p_rel_multiplier}")
    print(f"{'='*60}")

    params = {
        "version": "SynPopSim_MP_v1",
        "sim_code": sim_code,
        "axon_length": 300,
        "ps_spacing": 3,
        "num_slow_SVs": 10,
        "num_fast_SVs": 10,
        "time_step_size": 2,
        "time_steps": 9000,
        "time_interval": 0.3,
        "StartingPoolSize": 40,
        "p_ant": 0.50,
        "p_slow": p_slow,
        "p_cap_slow": 0.75,
        "p_cap_fast_ant": 0.75,
        "p_cap_fast_ret": p_cap_fast_ret,
        "svp_cap": 0.01,
        "faststep_multiplier": 1.5,
        "slowstep_multiplier": 0.5,
        "p_rel_multiplier": p_rel_multiplier,
        "rate_bg_multiplier": 4,
        "sim_date": str(date.today()),
        "notes": f"Parameter sweep run {sim_code}"
    }

    dt = params["time_step_size"]
    faststep = params["faststep_multiplier"] * dt
    slowstep = params["slowstep_multiplier"] * dt
    p_rel = params["p_rel_multiplier"] * dt
    rate_bg = int(params["rate_bg_multiplier"] * dt)

    ps_positions = np.arange(0, params["axon_length"] + params["ps_spacing"], params["ps_spacing"])
    num_sites = len(ps_positions)
    ps_set = set(ps_positions)
    spacing = params["ps_spacing"]
    axon_len = params["axon_length"]

    n_steps = params["time_steps"]
    pools = np.zeros([num_sites, n_steps], dtype=int)
    moving = np.zeros(n_steps)
    recycled = np.zeros_like(moving)
    totals = np.zeros_like(moving)
    SVPs = np.zeros_like(moving)

    def random_direction():
        return 1 if rnd.random() > params["p_ant"] else -1

    def make_sv(pos, sv_type, direction, state='moving'):
        return {'pos': float(pos), 'type': sv_type, 'dir': direction, 'state': state}

    # Initialize
    SVs = []
    for _ in range(params["num_slow_SVs"]):
        SVs.append(make_sv(rnd.randint(0, axon_len), 'slow', random_direction()))
    for _ in range(params["num_fast_SVs"]):
        SVs.append(make_sv(rnd.randint(0, axon_len), 'fast', random_direction()))
    for pos in ps_positions:
        for _ in range(params["StartingPoolSize"]):
            SVs.append(make_sv(pos, 'fast', 1, state='pooled'))

    # Simulation loop
    for t in range(n_steps):
        pct = 1 + 100 * t // n_steps
        print(f"  {sim_code}: {pct:03d}% complete", end='\r', flush=True)

        for _ in range(rate_bg):
            SVs.append(make_sv(0.0, 'fast', 1, state='SVP'))

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

        for sv in SVs:
            if sv['state'] == 'moving':
                step = faststep if sv['type'] == 'fast' else slowstep
                newpos = sv['pos'] + step * sv['dir']
                if newpos > axon_len:
                    newpos = axon_len - (newpos - axon_len)
                    sv['dir'] = -1
                elif newpos < 0:
                    sv['pos'] = 0
                    sv['state'] = 'recycled'
                    continue
                else:
                    sv['pos'] = newpos
                if sv['pos'] in ps_set:
                    if sv['type'] == 'slow':
                        p_cap = params["p_cap_slow"]
                    elif sv['dir'] == 1:
                        p_cap = params["p_cap_fast_ant"]
                    else:
                        p_cap = params["p_cap_fast_ret"]
                    if rnd.random() < p_cap:
                        sv['state'] = 'pooled'

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
                if newpos in ps_set:
                    if rnd.random() < params["svp_cap"]:
                        sv['state'] = 'pooled'

    print(f"  {sim_code}: 100% complete")

    # Save output
    export = np.row_stack((totals, SVPs, moving, recycled, pools))
    header = "Total SVs,SVPs,Axonal Transport SVs,Recycled SVs," + ",".join(f"PS Loc: {ps}" for ps in ps_positions)

    csv_path = os.path.join(OUTPUT_DIR, f"{sim_code}_output.csv")
    params_path = os.path.join(OUTPUT_DIR, f"{sim_code}_parameters.json")

    np.savetxt(csv_path, export.T, delimiter=",", fmt="%d", header=header, comments="")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

    print(f"  Saved: {csv_path}")
    print(f"  Saved: {params_path}")

    return pools, moving, recycled, SVPs, totals, ps_positions, params


def generate_figures(pools, moving, recycled, SVPs, totals, ps_positions, params, sim_code):
    """Generate and save figures for one simulation run."""
    pool_cols_count = pools.shape[0]
    n_steps = pools.shape[1]
    dx = params["ps_spacing"]
    dt_sec = params["time_step_size"]
    positions_um = np.arange(pool_cols_count) * dx
    time_hours = np.arange(n_steps) * dt_sec / 3600.0

    hour_steps = int(60 * 60 / dt_sec)
    t_max = min(int(5 * hour_steps), n_steps)
    pool_dists = pools[:, :t_max].T
    x_min, x_max = 0, (pool_cols_count - 1) * dx

    plt.style.use('seaborn-v0_8-whitegrid')
    COLORS = ['#2D6A4F', '#E76F51', '#264653', '#E9C46A']

    label = f"{sim_code} (ret={params['p_cap_fast_ret']}, rel={params['p_rel_multiplier']})"

    # 1. Spatiotemporal heatmap
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(pool_dists, aspect='auto', origin='lower',
                   extent=[x_min, x_max, 0, 5],
                   interpolation='bilinear', cmap='inferno')
    ax.set_yticks(range(0, 6))
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label('Vesicle Count', fontsize=11)
    ax.set_xlabel('Distance Along Axon (micrometers)', fontsize=11)
    ax.set_ylabel('Time (hours)', fontsize=11)
    ax.set_title(f'Pooled SV Distribution -- {label}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"{sim_code}_heatmap.png"), dpi=180, bbox_inches='tight')
    plt.close(fig)

    # 2. Final distribution
    final_dist = pool_dists[-1, :]
    fig, ax = plt.subplots(figsize=(11, 4))
    markerline, stemlines, baseline = ax.stem(positions_um, final_dist, linefmt=COLORS[2], markerfmt='o', basefmt='gray')
    plt.setp(stemlines, linewidth=0.8, alpha=0.7)
    plt.setp(markerline, markersize=3, color=COLORS[0])
    ax.set_xlabel('Distance Along Axon (micrometers)', fontsize=11)
    ax.set_ylabel('Vesicle Count', fontsize=11)
    ax.set_title(f'Final SV Distribution (5 hrs) -- {label}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"{sim_code}_final_dist.png"), dpi=180, bbox_inches='tight')
    plt.close(fig)

    # 3. Center of mass
    pool_sums = pool_dists.sum(axis=1)
    pool_sums[pool_sums == 0] = 1
    mean_pos = (pool_dists @ positions_um) / pool_sums
    time_hrs_sub = np.arange(t_max) * dt_sec / 3600.0

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_hrs_sub, mean_pos, color=COLORS[1], linewidth=1.5)
    ax.axhline(y=150, color='gray', linestyle='--', linewidth=0.8, alpha=0.6, label='Axon midpoint (150 um)')
    ax.set_xlabel('Time (hours)', fontsize=11)
    ax.set_ylabel('Mean Position (micrometers)', fontsize=11)
    ax.set_ylim(0, 300)
    ax.set_title(f'Center of Mass -- {label}', fontsize=13, fontweight='bold')
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"{sim_code}_center_mass.png"), dpi=180, bbox_inches='tight')
    plt.close(fig)

    # 4. Population dynamics
    pooled_total = pools.sum(axis=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_hours, moving, color=COLORS[0], linewidth=1.5, label='Moving')
    ax.plot(time_hours, recycled, color=COLORS[1], linewidth=1.5, label='Recycled')
    ax.plot(time_hours, pooled_total, color=COLORS[2], linewidth=1.5, label='Pooled')
    ax.plot(time_hours, SVPs, color=COLORS[3], linewidth=1.5, label='SVPs')
    ax.set_xlabel('Time (hours)', fontsize=11)
    ax.set_ylabel('Vesicle Count', fontsize=11)
    ax.set_title(f'Population Dynamics -- {label}', fontsize=13, fontweight='bold')
    ax.legend(frameon=True, fontsize=9, loc='upper left')
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f"{sim_code}_populations.png"), dpi=180, bbox_inches='tight')
    plt.close(fig)

    print(f"  Figures saved for {sim_code}")


if __name__ == "__main__":
    print(f"Starting batch run: {len(RUNS)} simulations")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Figures directory: {FIGURES_DIR}")

    for i, run in enumerate(RUNS):
        sim_code = run["sim_code"]
        print(f"\n[{i+1}/{len(RUNS)}] Running {sim_code}...")

        pools, moving, recycled, SVPs, totals, ps_positions, params = run_simulation(
            p_cap_fast_ret=run["p_cap_fast_ret"],
            p_slow=run["p_slow"],
            p_rel_multiplier=run["p_rel_multiplier"],
            sim_code=sim_code
        )

        generate_figures(pools, moving, recycled, SVPs, totals, ps_positions, params, sim_code)

    print(f"\n{'='*60}")
    print(f"ALL DONE -- {len(RUNS)} runs completed")
    print(f"Output files: {OUTPUT_DIR}")
    print(f"Figure files: {FIGURES_DIR}")
    print(f"{'='*60}")
