#!/usr/bin/env python3
"""
Compute center-of-mass distance between a peptide and a protein pocket
over an MD trajectory using MDAnalysis.

USAGE
-----
python com_distance.py \
    --gro md_100ns.gro \
    --xtc md_100ns.xtc \
    --peptide-range 358-370 \
    --pocket-range 155-163 \
    --out com_distance.dat \
    --interval 0.5

ARGUMENTS
---------
--gro            Path to topology (.gro)
--xtc            Path to trajectory (.xtc)
--peptide-range  MDAnalysis resindex range, e.g. 358-370
--pocket-range   Protein resid range, e.g. 155-163
--out            Output file for distances
--interval       Desired sampling interval in ns (approx.)
"""

import argparse
import numpy as np
import MDAnalysis as mda


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute COM distance between peptide and protein pocket."
    )

    p.add_argument("--gro", required=True,
                   help="Topology file (.gro), REQUIRED")
    p.add_argument("--xtc", required=True,
                   help="Trajectory file (.xtc), REQUIRED")

    p.add_argument("--peptide-range", required=True,
                   help="Peptide resindex range, e.g. 358-370 (MDAnalysis indices)")
    p.add_argument("--pocket-range", required=True,
                   help="Pocket resid range, e.g. 155-163 (GROMACS residue IDs)")

    p.add_argument("--out", required=True,
                   help="Output file for COM distances")

    p.add_argument("--interval", required=True, type=float,
                   help="Desired sampling interval between analyzed frames (ns)")

    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading universe from {args.gro} and {args.xtc} ...")
    u = mda.Universe(args.gro, args.xtc)

    # --- Build selections ---
    peptide_sel = f"resindex {args.peptide_range}"
    pocket_sel = f"protein and resid {args.pocket_range}"

    peptide = u.select_atoms(peptide_sel)
    pocket = u.select_atoms(pocket_sel)

    print(f"Peptide selection: '{peptide_sel}' -> {peptide.n_atoms} atoms")
    print(f"Pocket  selection: '{pocket_sel}' -> {pocket.n_atoms} atoms")

    if peptide.n_atoms == 0:
        raise RuntimeError("Peptide selection is empty. Check peptide resindex range.")
    if pocket.n_atoms == 0:
        raise RuntimeError("Pocket selection is empty. Check pocket resid range.")

    # --- Determine stride ---
    dt_ps = u.trajectory.dt
    if dt_ps is None or dt_ps <= 0:
        print("Trajectory dt not available; analyzing every frame.")
        stride = 1
    else:
        desired_ps = args.interval * 1000.0
        stride = max(1, int(round(desired_ps / dt_ps)))
        print(f"Trajectory dt = {dt_ps:.3f} ps -> stride = {stride} "
              f"(~{stride * dt_ps / 1000.0:.3f} ns between frames)")

    times_ns = []
    distances_A = []

    print("Starting COM distance calculation ...")
    for ts in u.trajectory[::stride]:
        t_ns = ts.time / 1000.0
        com_pep = peptide.center_of_mass(pbc=True)
        com_poc = pocket.center_of_mass(pbc=True)

        d = np.linalg.norm(com_pep - com_poc)

        times_ns.append(t_ns)
        distances_A.append(d)

    times_ns = np.array(times_ns)
    distances_A = np.array(distances_A)

    print(f"Writing distances to {args.out} ...")
    with open(args.out, "w") as f:
        f.write("# time_ns distance_A\n")
        for t, d in zip(times_ns, distances_A):
            f.write(f"{t:12.5f} {d:12.5f}\n")

    print("Done.")
    print(f"Frames analyzed: {len(times_ns)}")
    print(f"Distance (Å): min={distances_A.min():.3f}, "
          f"max={distances_A.max():.3f}, mean={distances_A.mean():.3f}")


if __name__ == "__main__":
    main()
