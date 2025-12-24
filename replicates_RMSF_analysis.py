
#!/usr/bin/env python3
import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.rms import RMSF


# --------------------------
# CLI
# --------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Generate per-replicate, overlay, and averaged RMSF plots"
    )

    p.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Root directory containing replicate folders",
    )
    p.add_argument(
        "--rep-glob",
        default="rep*",
        help="Glob for replicate directories (relative to --root)",
    )
    p.add_argument(
        "--trajectory-pattern",
        default="*.xtc",
        help="Glob for trajectory files inside replicate folder",
    )
    p.add_argument(
        "--topology-patterns",
        default="*.pdb,*.psf,*.gro,*.tpr,*.prmtop",
        help="Comma-separated topology filename globs (searched inside replicate folder)",
    )
    p.add_argument(
        "--selection",
        default="protein and backbone",
        help="MDAnalysis atom selection for RMSF (e.g., 'protein and backbone')",
    )
    p.add_argument(
        "--per-rep-dir",
        default="rmsf_per_rep",
        help="Directory name for per-replicate plots (created inside each replicate folder)",
    )
    p.add_argument(
        "--overlay-dir",
        default="rmsf_overlay",
        help="Directory name for overlay/average plots (created under --root)",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI",
    )
    p.add_argument(
        "--split-mode",
        choices=["resid_resets", "none"],
        default="resid_resets",
        help="How to split chains. 'resid_resets' splits when residue numbering resets (resid decreases).",
    )

    return p.parse_args()


# --------------------------
# Helpers
# --------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "unnamed"


def _close_universe(universe: "mda.Universe") -> None:
    traj = getattr(universe, "trajectory", None)
    if traj is not None and hasattr(traj, "close"):
        try:
            traj.close()
        except Exception:
            pass


@dataclass(frozen=True)
class Replicate:
    name: str
    root: Path
    topology: Path
    trajectory: Path


def _first_match(folder: Path, globs: Sequence[str]) -> Path:
    for pat in globs:
        hits = sorted(folder.glob(pat))
        if hits:
            return hits[0]
    return None  # type: ignore


def discover_replicates(
    root: Path,
    rep_glob: str,
    trajectory_pattern: str,
    topology_globs: Sequence[str],
) -> List[Replicate]:
    reps: List[Replicate] = []

    for rep_dir in sorted(root.glob(rep_glob)):
        if not rep_dir.is_dir():
            continue

        topo = _first_match(rep_dir, topology_globs)
        trajs = sorted(rep_dir.glob(trajectory_pattern))
        traj = trajs[0] if trajs else None

        if topo is None or traj is None:
            continue

        reps.append(
            Replicate(
                name=rep_dir.name,
                root=rep_dir,
                topology=topo,
                trajectory=traj,
            )
        )

    return reps


def split_by_resid_resets(
    ag: "mda.core.groups.AtomGroup",
) -> List[Tuple[str, "mda.core.groups.AtomGroup"]]:
    """
    Split a protein AtomGroup into chains by detecting residue numbering resets.

    Logic: iterate residues in order; whenever resids[i] <= resids[i-1], start a new chain.
    This matches cases like:
      ... 505, 506, 507, 1, 2, 3, ...
    """
    residues = ag.residues
    if len(residues) == 0:
        return []

    resids = np.asarray(residues.resids, dtype=int)

    # Find breakpoints where numbering decreases or repeats/backtracks
    breaks = [0]
    for i in range(1, len(resids)):
        if resids[i] <= resids[i - 1]:
            breaks.append(i)
    breaks.append(len(resids))

    # Build segments
    segments: List[Tuple[str, "mda.core.groups.AtomGroup"]] = []
    chain_num = 1
    for start, end in zip(breaks[:-1], breaks[1:]):
        seg_res = residues[start:end]
        if len(seg_res) == 0:
            continue

        seg_atoms = seg_res.atoms
        if len(seg_atoms) == 0:
            continue

        lo = int(np.min(seg_res.resids))
        hi = int(np.max(seg_res.resids))
        label = f"chain{chain_num}_resid{lo}-{hi}"
        segments.append((label, seg_atoms))
        chain_num += 1

    return segments


# --------------------------
# RMSF + Plotting
# --------------------------
def compute_chain_rmsf(
    universe: "mda.Universe",
    selection: str,
    split_mode: str = "resid_resets",
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Returns list of (chain_label, residue_indices, rmsf_values).

    RMSF computed at atom level then averaged per residue.
    """
    sel = universe.select_atoms(selection)
    if len(sel) == 0:
        raise ValueError(f"Selection matched 0 atoms: {selection}")

    if split_mode == "resid_resets":
        chain_groups = split_by_resid_resets(sel)
        if not chain_groups:
            chain_groups = [("ALL", sel)]
    else:
        chain_groups = [("ALL", sel)]

    results: List[Tuple[str, np.ndarray, np.ndarray]] = []

    for chain_label, ag in chain_groups:
        # Atom-level RMSF
        r = RMSF(ag).run()
        atom_rmsf = np.asarray(r.rmsf, dtype=float)

        # Residue mapping for these atoms
        atom_resids = np.asarray([a.resid for a in ag.atoms], dtype=int)
        unique_res = np.unique(atom_resids)

        # Average atom RMSF per residue
        per_res = np.zeros_like(unique_res, dtype=float)
        for i, resid in enumerate(unique_res):
            mask = (atom_resids == resid)
            per_res[i] = atom_rmsf[mask].mean() if np.any(mask) else np.nan

        results.append((chain_label, unique_res, per_res))

    return results


def plot_chain(
    rep_name: str,
    chain_label: str,
    residues: Sequence[int],
    values: Sequence[float],
    path: Path,
    dpi: int,
) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(residues, values, lw=1.5)
    plt.xlabel("Residue index")
    plt.ylabel("RMSF (Å)")
    plt.title(f"{rep_name} – {chain_label}")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def plot_overlay(
    chain_label: str,
    series: Sequence[Tuple[str, Sequence[int], Sequence[float]]],
    path: Path,
    dpi: int,
) -> None:
    plt.figure(figsize=(8, 4))
    for name, residues, values in sorted(series, key=lambda item: item[0]):
        plt.plot(residues, values, lw=1.5, label=name)
    plt.xlabel("Residue index")
    plt.ylabel("RMSF (Å)")
    plt.title(f"{chain_label} – replicates overlay")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def plot_average(
    chain_label: str,
    series: Sequence[Tuple[str, Sequence[int], Sequence[float]]],
    path: Path,
    dpi: int,
    *,
    shade_std: bool = True,
) -> None:
    if not series:
        return

    residue_sets = [set(map(int, residues)) for _, residues, _ in series]
    common = sorted(set.intersection(*residue_sets)) if residue_sets else []
    if not common:
        raise ValueError(f"No common residues found across replicates for '{chain_label}'")

    common_arr = np.asarray(common, dtype=int)

    stacked = []
    for _, residues, values in series:
        r = np.asarray(residues, dtype=int)
        v = np.asarray(values, dtype=float)
        mapping = {int(rr): float(vv) for rr, vv in zip(r, v)}
        stacked.append(np.asarray([mapping[int(rr)] for rr in common_arr], dtype=float))

    mat = np.vstack(stacked)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros_like(mean)

    plt.figure(figsize=(8, 4))
    plt.plot(common_arr, mean, lw=2.0, label="Mean")
    if shade_std and mat.shape[0] > 1:
        plt.fill_between(common_arr, mean - std, mean + std, alpha=0.2, label="±1 SD")

    plt.xlabel("Residue index")
    plt.ylabel("RMSF (Å)")
    plt.title(f"{chain_label} – mean RMSF ({mat.shape[0]} reps)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


# --------------------------
# Main
# --------------------------
def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    topology_globs = [item.strip() for item in args.topology_patterns.split(",") if item.strip()]

    replicates = discover_replicates(root, args.rep_glob, args.trajectory_pattern, topology_globs)
    if not replicates:
        raise SystemExit("No replicate folders with usable topology/trajectory were found.")

    overlay_bucket: Dict[str, List[Tuple[str, Sequence[int], Sequence[float]]]] = defaultdict(list)
    overlay_root = root / args.overlay_dir
    ensure_dir(overlay_root)

    for replicate in replicates:
        print(f"[info] Processing {replicate.name} …")
        universe = mda.Universe(str(replicate.topology), str(replicate.trajectory))
        try:
            chain_data = compute_chain_rmsf(universe, args.selection, split_mode=args.split_mode)
        finally:
            _close_universe(universe)

        per_rep_dir = replicate.root / args.per_rep_dir
        ensure_dir(per_rep_dir)

        for chain_label, residues, rmsf_values in chain_data:
            overlay_bucket[chain_label].append((replicate.name, residues, rmsf_values))

            safe_chain = _safe_slug(chain_label)
            safe_rep = _safe_slug(replicate.name)
            filename = f"{safe_rep}_{safe_chain}_rmsf.png"

            plot_chain(
                replicate.name,
                chain_label,
                residues,
                rmsf_values,
                per_rep_dir / filename,
                args.dpi,
            )

        print(f"[info] Saved {len(chain_data)} chain plots under {per_rep_dir}")

    # Overlay + mean plots (per chain segment label)
    for chain_label, series in overlay_bucket.items():
        safe_chain = _safe_slug(chain_label)

        overlay_out = overlay_root / f"{safe_chain}_overlay.png"
        plot_overlay(chain_label, series, overlay_out, args.dpi)

        avg_out = overlay_root / f"{safe_chain}_average.png"
        plot_average(chain_label, series, avg_out, args.dpi)

    print(f"[info] Overlay + average plots saved to {overlay_root}")


if __name__ == "__main__":
    main()
