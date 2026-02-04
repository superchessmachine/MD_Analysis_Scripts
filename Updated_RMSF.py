#!/usr/bin/env python3
"""
RMSF per chain (MDAnalysis) using atom selections (no manual CA index ranges).
This script is newer because it adds a list feature before you run it
Typical usage (for your folder):
  python rmsf_two_chains.py --top md_30ns.tpr --traj md_30ns_fit_dt500ps.xtc --list-chains

Then run (example selectors that often work with GROMACS .tpr):
  python rmsf_two_chains.py \
    --top md_30ns.tpr \
    --traj md_30ns_fit_dt500ps.xtc \
    --chain "A:segid Protein_chain_A" \
    --chain "B:segid Protein_chain_B" \
    --outdir rmsf_plots

Outputs:
  - rmsf_overlay.png  (both chains on one plot)
  - rmsf_A.png, rmsf_B.png (per-chain plots)
  - rmsf_values.tsv   (table of resid + RMSF per chain)
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
    Parse --chain "NAME:SELECTION"
    Example: "A:segid Protein_chain_A"
    """
    if ":" not in chain_arg:
        raise ValueError(f'Invalid --chain "{chain_arg}". Expected "NAME:SELECTION".')
    name, sel = chain_arg.split(":", 1)
    name = name.strip()
    sel = sel.strip()
    if not name or not sel:
        raise ValueError(f'Invalid --chain "{chain_arg}". Expected "NAME:SELECTION".')
    return name, sel


def list_chain_hints(u):
    # segids
    segids = sorted({s.segid for s in u.segments if getattr(s, "segid", "")})
    chainids = sorted({a.chainID for a in u.atoms if getattr(a, "chainID", "")})

    print("\n=== MDAnalysis chain hints ===")
    print(f"Segments found: {len(u.segments)}")
    if segids:
        print("Unique segid values:")
        for s in segids:
            print(f"  - {s}")
    else:
        print("No non-empty segid values found.")

    if chainids:
        print("Unique chainID values:")
        for c in chainids:
            print(f"  - {c}")
    else:
        print("No non-empty chainID values found.")

    print("\nCommon selectors to try:")
    if segids:
        print('  --chain "A:segid <one_of_the_segids_above>"')
    if chainids:
        print('  --chain "A:chainID <one_of_the_chainIDs_above>"')
    print('  Or inspect with: u.atoms[:10] in a python shell.\n')


def compute_rmsf_for_selection(u, selection, align_sel="protein and name CA"):
    """
    Align trajectory to itself (first frame reference) using align_sel,
    then compute RMSF for Cα atoms in `selection`.
    """
    # align all frames to the first frame reference
    align.AlignTraj(u, u, select=align_sel, in_memory=True).run()

    ca = u.select_atoms(f"({selection}) and protein and name CA")
    if ca.n_atoms == 0:
        raise RuntimeError(f'Selection produced 0 Cα atoms: ({selection}) and protein and name CA')

    r = rms.RMSF(ca).run()

    resids = ca.resids.astype(int)
    rmsf_vals = r.results.rmsf

    # Collapse duplicates (safe if something odd happens with selection)
    bucket = {}
    for resid, val in zip(resids, rmsf_vals):
        bucket.setdefault(resid, []).append(float(val))

    resid_arr = np.array(sorted(bucket.keys()), dtype=int)
    rmsf_arr = np.array([np.mean(bucket[r]) for r in resid_arr], dtype=float)
    return resid_arr, rmsf_arr


def plot_single(resids, rmsf_vals, title, outpath):
    plt.figure(figsize=(10, 4.5))
    plt.plot(resids, rmsf_vals, linewidth=2)
    plt.xlabel("Residue ID")
    plt.ylabel("RMSF (Å)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_overlay(series, outpath):
    """
    series = list of (name, resids, rmsf_vals)
    """
    plt.figure(figsize=(11, 5))
    for name, resids, rmsf_vals in series:
        plt.plot(resids, rmsf_vals, linewidth=2, label=name)
    plt.xlabel("Residue ID")
    plt.ylabel("RMSF (Å)")
    plt.title("RMSF (Cα) — overlay")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def write_tsv(series, outpath):
    """
    Writes a wide table:
      resid   RMSF_<chain1>   RMSF_<chain2> ...
    Missing values are blank.
    """
    all_resids = sorted(set().union(*[set(resids.tolist()) for _, resids, _ in series]))
    name_to_map = {}
    for name, resids, rmsf_vals in series:
        name_to_map[name] = {int(r): float(v) for r, v in zip(resids, rmsf_vals)}

    names = [name for name, _, _ in series]
    with open(outpath, "w") as f:
        f.write("resid\t" + "\t".join([f"RMSF_{n}" for n in names]) + "\n")
        for r in all_resids:
            row = [str(r)]
            for n in names:
                val = name_to_map[n].get(r, None)
                row.append("" if val is None else f"{val:.4f}")
            f.write("\t".join(row) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Compute RMSF per chain using MDAnalysis selections.")
    ap.add_argument("--top", required=True, help="Topology (.tpr)")
    ap.add_argument("--traj", required=True, help="Trajectory (.xtc)")
    ap.add_argument(
        "--chain",
        action="append",
        help='Chain definition "NAME:SELECTION". Repeat for multiple chains.'
    )
    ap.add_argument("--outdir", default="rmsf_plots", help="Output directory")
    ap.add_argument(
        "--align-sel",
        default="protein and name CA",
        help='Selection to align on (default: "protein and name CA")'
    )
    ap.add_argument(
        "--list-chains",
        action="store_true",
        help="Print segid/chainID hints and exit."
    )

    args = ap.parse_args()

    u = mda.Universe(args.top, args.traj)

    if args.list_chains:
        list_chain_hints(u)
        return

    if not args.chain or len(args.chain) < 2:
        raise SystemExit('Provide at least two --chain entries, e.g. --chain "A:segid X" --chain "B:segid Y"')

    os.makedirs(args.outdir, exist_ok=True)

    series = []
    for chain_arg in args.chain:
        name, sel = parse_chain_arg(chain_arg)
        print(f"Computing RMSF for {name} with selection: {sel}")
        resids, rmsf_vals = compute_rmsf_for_selection(u, sel, align_sel=args.align_sel)
        series.append((name, resids, rmsf_vals))

        outpath = os.path.join(args.outdir, f"rmsf_{name}.png")
        plot_single(resids, rmsf_vals, f"RMSF (Cα) — {name}", outpath)
        print(f"Wrote: {outpath}")

    overlay_path = os.path.join(args.outdir, "rmsf_overlay.png")
    plot_overlay(series, overlay_path)
    print(f"Wrote: {overlay_path}")

    tsv_path = os.path.join(args.outdir, "rmsf_values.tsv")
    write_tsv(series, tsv_path)
    print(f"Wrote: {tsv_path}")


if __name__ == "__main__":
    main()
