#!/usr/bin/env python3
"""
USAGE EXAMPLE (READ THIS FIRST):

This script generates per-chain RMSF plots using Cα atoms, where
YOU specify each chain by NAME and by a RANGE OF INDICES.

IMPORTANT:
- All indices you provide are **1-based and inclusive**
  (i.e. the first Cα atom is index 1, not 0).
- You can specify multiple chains by repeating --chain.
- One RMSF plot is generated per chain automatically.

EXAMPLE COMMAND:

python rmsf_per_chain.py \
  --top md_30ns_4fs.tpr \
  --traj 30ns_fit_dt500ps.xtc \
  --chain CHAINA:1:58 \
  --chain CHAINB:59:429 \
  --chain CHAINC:430:443 \
  --outdir rmsf_plots

WHAT EACH FLAG MEANS:

--top
  Path to the GROMACS topology file (.tpr).

--traj
  Path to the trajectory file (.xtc).

--chain NAME:START:END
  Defines one chain to analyze.
  - NAME  = label used in plot title and filename
  - START = first Cα index (1-based, inclusive)
  - END   = last Cα index (1-based, inclusive)
  This flag can be repeated multiple times for multiple chains.

--outdir
  Directory where RMSF plots will be written.
  (Default: rmsf_plots)

OUTPUT:

For each --chain entry, the script produces:
  rmsf_<CHAIN_NAME>.png

Example outputs:
  rmsf_ATXN7L3.png
  rmsf_USP27X.png
  rmsf_Linear_hD2.png
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import MDAnalysis as mda
from MDAnalysis.analysis import align, rms


def parse_chain_arg(chain_arg):
    """
    Parse chain argument of the form:
      name:start:end
    where start/end are 1-based inclusive indices.
    """
    try:
        name, start, end = chain_arg.split(":")
        start = int(start)
        end = int(end)
        if start < 1 or end < 1 or end < start:
            raise ValueError
        return name, start, end
    except Exception:
        raise ValueError(
            f"Invalid --chain format: {chain_arg}\n"
            "Expected: name:start:end (1-based indices)"
        )


def compute_rmsf(top, traj, start_1based, end_1based):
    # Convert to 0-based indices for Python slicing
    start = start_1based - 1
    end = end_1based - 1

    u = mda.Universe(top, traj)

    ca_all = u.select_atoms("protein and name CA")
    if ca_all.n_atoms <= end:
        raise RuntimeError(
            f"Only {ca_all.n_atoms} CA atoms found, "
            f"but requested up to index {end_1based}"
        )

    ca_sel = ca_all[start:end + 1]

    # Align on all protein Cα atoms to remove global motion
    align.AlignTraj(
        u,
        u,
        select="protein and name CA",
        in_memory=True
    ).run()

    r = rms.RMSF(ca_sel).run()

    resids = ca_sel.resids.astype(int)
    rmsf_vals = r.results.rmsf

    # Collapse by residue ID (handles duplicates safely)
    bucket = {}
    for resid, val in zip(resids, rmsf_vals):
        bucket.setdefault(resid, []).append(val)

    resid_arr = np.array(sorted(bucket.keys()))
    rmsf_arr = np.array([np.mean(bucket[r]) for r in resid_arr])

    return resid_arr, rmsf_arr


def plot_rmsf(resids, rmsf, chain_name, outdir):
    plt.figure(figsize=(10, 4.5))
    plt.plot(resids, rmsf, linewidth=2)
    plt.xlabel("Residue ID")
    plt.ylabel("RMSF (Å)")
    plt.title(f"RMSF (Cα) — {chain_name}")
    plt.tight_layout()

    outpath = os.path.join(outdir, f"rmsf_{chain_name}.png")
    plt.savefig(outpath, dpi=200)
    plt.close()

    print(f"Wrote: {outpath}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-chain RMSF plots using 1-based Cα indices"
    )

    parser.add_argument(
        "--top",
        required=True,
        help="Topology file (.tpr)"
    )
    parser.add_argument(
        "--traj",
        required=True,
        help="Trajectory file (.xtc)"
    )
    parser.add_argument(
        "--chain",
        action="append",
        required=True,
        help="Chain definition: NAME:START:END (1-based indices). "
             "Repeat for multiple chains."
    )
    parser.add_argument(
        "--outdir",
        default="rmsf_plots",
        help="Output directory (default: rmsf_plots)"
    )

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    for chain_arg in args.chain:
        name, start, end = parse_chain_arg(chain_arg)
        print(f"Processing {name}: indices {start}–{end} (1-based)")

        resids, rmsf_vals = compute_rmsf(
            args.top, args.traj, start, end
        )

        plot_rmsf(resids, rmsf_vals, name, args.outdir)


if __name__ == "__main__":
    main()
