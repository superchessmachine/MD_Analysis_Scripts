#!/usr/bin/env python3
"""
Automated RMSF comparison plus quick GBSA/PBSA interaction analysis.

This workflow was written for packaged GROMACS "view" trajectories that contain
only .gro/.xtc files. It does two things:

1. Computes aligned protein C-alpha RMSF for two simulations:
   - full trajectory
   - last N ns
   Outputs PNG overlays and CSV tables.

2. Optionally runs quick protein-ligand GBSA and/or PBSA interaction analyses
   for the ligand-bound system. It first tries to use gmx_MMPBSA for structure
   cleanup, then builds Amber prmtops with tleap and runs AmberTools MMPBSA.py
   on a stripped protein+ligand trajectory.

Example:
  python md_rmsf_mmpbsa_workflow.py \\
    --root /Users/ysb/analysis_project \\
    --apo-gro apo_monomer/apo_monomer_view.gro \\
    --apo-xtc apo_monomer/apo_monomer_view.xtc \\
    --bound-gro monomer_ligand_rep1/monomer_ligand_rep1_view.gro \\
    --bound-xtc monomer_ligand_rep1/monomer_ligand_rep1_view.xtc \\
    --ligand-resname UNL \\
    --ligand-charge 1 \\
    --run-gbsa \\
    --run-pbsa
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.analysis import align, rms


@dataclass(frozen=True)
class SystemSpec:
    label: str
    gro: Path
    xtc: Path
    color: str


def run_command(
    cmd: list[str],
    cwd: Path,
    input_text: str | None = None,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    print("[CMD]", " ".join(str(c) for c in cmd))
    result = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd),
        input=input_text,
        text=True,
        capture_output=True,
        env=env,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(str(c) for c in cmd)}")
    return result


def tool_env(amber_bin: Path, gmx_exe: str | Path) -> dict[str, str]:
    """Build an environment where AmberTools and GROMACS helper programs resolve."""
    env = os.environ.copy()
    gmx_dir = Path(gmx_exe).resolve().parent
    env["PATH"] = os.pathsep.join([str(amber_bin), str(gmx_dir), env.get("PATH", "")])
    env.setdefault("AMBERHOME", str(amber_bin.parent))
    env.setdefault("GMX_MAXBACKUP", "-1")
    return env


def find_executable(default: str, fallback: str | None = None) -> str:
    found = shutil.which(default)
    if found:
        return found
    if fallback and Path(fallback).exists():
        return fallback
    return default


def trajectory_info(system: SystemSpec) -> dict[str, float | int]:
    universe = mda.Universe(str(system.gro), str(system.xtc))
    times = np.array([ts.time for ts in universe.trajectory], dtype=float)
    protein = universe.select_atoms("protein")
    ca = universe.select_atoms("protein and name CA")
    return {
        "atoms": len(universe.atoms),
        "residues": len(universe.residues),
        "protein_atoms": len(protein),
        "protein_residues": len(protein.residues),
        "ca_atoms": len(ca),
        "frames": len(universe.trajectory),
        "first_time_ps": float(times[0]),
        "last_time_ps": float(times[-1]),
        "dt_ps": float(np.median(np.diff(times))) if len(times) > 1 else 0.0,
        "duration_ns": float((times[-1] - times[0]) / 1000.0),
    }


def frame_index_at_or_after(gro: Path, xtc: Path, time_ps: float) -> int:
    universe = mda.Universe(str(gro), str(xtc))
    times = np.array([ts.time for ts in universe.trajectory], dtype=float)
    index = int(np.searchsorted(times, time_ps, side="left"))
    return min(index, len(times) - 1)


def aligned_ca_rmsf(gro: Path, xtc: Path, start_frame: int = 0) -> pd.DataFrame:
    universe = mda.Universe(str(gro), str(xtc))
    reference = mda.Universe(str(gro), str(xtc))
    ca = universe.select_atoms("protein and name CA")
    if len(ca) == 0:
        raise ValueError(f"No protein C-alpha atoms found in {gro}")

    align.AlignTraj(universe, reference, select="protein and name CA", in_memory=True).run(start=start_frame)
    values = rms.RMSF(ca).run(start=start_frame).results.rmsf
    return pd.DataFrame(
        {
            "resid": ca.resids,
            "resname": ca.resnames,
            "residue": [f"{resname}{resid}" for resname, resid in zip(ca.resnames, ca.resids)],
            "rmsf_A": values,
        }
    )


def plot_rmsf(data: dict[str, pd.DataFrame], systems: list[SystemSpec], title: str, outfile: Path) -> None:
    colors = {system.label: system.color for system in systems}
    fig, ax = plt.subplots(figsize=(12, 4.8), constrained_layout=True)
    for label, df in data.items():
        ax.plot(df["resid"], df["rmsf_A"], lw=1.7, label=label, color=colors.get(label))
    ax.set_title(title)
    ax.set_xlabel("Residue number")
    ax.set_ylabel("C-alpha RMSF (Angstrom)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(outfile, dpi=300)
    plt.close(fig)


def run_rmsf(systems: list[SystemSpec], outdir: Path, last_ns: float) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    infos = {system.label: trajectory_info(system) for system in systems}
    full: dict[str, pd.DataFrame] = {}
    last: dict[str, pd.DataFrame] = {}

    for system in systems:
        info = infos[system.label]
        first_ps = float(info["first_time_ps"])
        last_ps = float(info["last_time_ps"])
        start_ps = max(first_ps, last_ps - last_ns * 1000.0)
        start_frame = frame_index_at_or_after(system.gro, system.xtc, start_ps)

        full[system.label] = aligned_ca_rmsf(system.gro, system.xtc)
        last[system.label] = aligned_ca_rmsf(system.gro, system.xtc, start_frame=start_frame)

        safe = system.label.lower().replace(" ", "_").replace("/", "_")
        full[system.label].to_csv(outdir / f"{safe}_full_rmsf.csv", index=False)
        last[system.label].to_csv(outdir / f"{safe}_last{last_ns:g}ns_rmsf.csv", index=False)

    if len(systems) == 2:
        left, right = systems
        for name, tables in [("full", full), (f"last{last_ns:g}ns", last)]:
            merged = tables[left.label].merge(
                tables[right.label],
                on=["resid", "resname", "residue"],
                how="inner",
                suffixes=(f"_{left.label}", f"_{right.label}"),
            )
            merged["delta_right_minus_left_A"] = merged[f"rmsf_A_{right.label}"] - merged[f"rmsf_A_{left.label}"]
            merged.to_csv(outdir / f"rmsf_comparison_{name}.csv", index=False)

    plot_rmsf(full, systems, "C-alpha RMSF comparison: full trajectory", outdir / "rmsf_comparison_full.png")
    plot_rmsf(last, systems, f"C-alpha RMSF comparison: last {last_ns:g} ns", outdir / f"rmsf_comparison_last{last_ns:g}ns.png")

    with (outdir / "rmsf_summary.txt").open("w") as handle:
        handle.write("RMSF analysis summary\n=====================\n\n")
        for label, info in infos.items():
            handle.write(f"{label}\n")
            for key, value in info.items():
                handle.write(f"  {key}: {value}\n")
            handle.write("\n")
    print(f"[OK] RMSF outputs written to {outdir}")


def parse_ndx_groups(path: Path) -> dict[str, int]:
    groups: dict[str, int] = {}
    current_index = -1
    with path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                current_index += 1
                groups[stripped.strip("[] ").strip()] = current_index
    return groups


def write_single_group_ndx(atom_indices_1based: np.ndarray, group_name: str, path: Path) -> None:
    with path.open("w") as handle:
        handle.write(f"[ {group_name} ]\n")
        for start in range(0, len(atom_indices_1based), 15):
            handle.write(" ".join(str(int(i)) for i in atom_indices_1based[start : start + 15]) + "\n")


def detect_disulfides(pdb_path: Path, cutoff_a: float = 2.35) -> list[tuple[int, int]]:
    sg_atoms: list[tuple[int, np.ndarray]] = []
    with pdb_path.open() as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            resname = line[17:20].strip()
            if atom_name != "SG" or resname != "CYX":
                continue
            resid = int(line[22:26])
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float)
            sg_atoms.append((resid, xyz))

    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    for i, (resid_i, xyz_i) in enumerate(sg_atoms):
        if resid_i in used:
            continue
        best: tuple[int, float] | None = None
        for resid_j, xyz_j in sg_atoms[i + 1 :]:
            if resid_j in used:
                continue
            dist = float(np.linalg.norm(xyz_i - xyz_j))
            if dist <= cutoff_a and (best is None or dist < best[1]):
                best = (resid_j, dist)
        if best is not None:
            pairs.append((resid_i, best[0]))
            used.update({resid_i, best[0]})
    return pairs


def write_gbsa_input(path: Path, startframe: int, endframe: int, interval: int, gmx_style: bool = False) -> None:
    general_extra = ""
    if gmx_style:
        general_extra = '  forcefields = "oldff/leaprc.ff99SB,leaprc.gaff"\n  PBRadii = 3\n'
    path.write_text(
        "&general\n"
        f"  startframe = {startframe}\n"
        f"  endframe = {endframe}\n"
        f"  interval = {interval}\n"
        f"{general_extra}"
        "  keep_files = 2\n"
        "  verbose = 1\n"
        "/\n\n"
        "&gb\n"
        "  igb = 5\n"
        "  saltcon = 0.150\n"
        "/\n"
    )


def write_pbsa_input(path: Path, startframe: int, endframe: int, interval: int) -> None:
    path.write_text(
        "&general\n"
        f"  startframe = {startframe}\n"
        f"  endframe = {endframe}\n"
        f"  interval = {interval}\n"
        "  keep_files = 2\n"
        "  verbose = 1\n"
        "/\n\n"
        "&pb\n"
        "  istrng = 0.150\n"
        "  fillratio = 4.0\n"
        "  radiopt = 0\n"
        "  inp = 1\n"
        "  cavity_surften = 0.005\n"
        "  cavity_offset = 0.000\n"
        "  sprob = 1.400\n"
        "/\n"
    )


def write_manual_leap(
    path: Path,
    rec_pdb: Path,
    ligand_mol2: Path,
    ligand_frcmod: Path,
    disulfides: list[tuple[int, int]],
    outdir: Path,
) -> None:
    bonds = "\n".join(f"bond REC_OUT.{a}.SG REC_OUT.{b}.SG" for a, b in disulfides)
    combonds = "\n".join(f"bond COM_OUT.{a}.SG COM_OUT.{b}.SG" for a, b in disulfides)
    path.write_text(
        "source oldff/leaprc.ff99SB\n"
        "source leaprc.gaff\n"
        "loadOff atomic_ions.lib\n"
        "loadamberparams frcmod.ions234lm_126_tip3p\n"
        "set default PBRadii mbondi2\n\n"
        f"REC1 = loadpdb {rec_pdb}\n"
        f"LIG1 = loadmol2 {ligand_mol2}\n"
        f"loadamberparams {ligand_frcmod}\n\n"
        f"saveamberparm LIG1 {outdir / 'LIG.prmtop'} {outdir / 'LIG.inpcrd'}\n\n"
        "REC_OUT = combine { REC1 }\n"
        f"{bonds}\n"
        f"saveamberparm REC_OUT {outdir / 'REC.prmtop'} {outdir / 'REC.inpcrd'}\n\n"
        "COM_OUT = combine { REC1 LIG1 }\n"
        f"{combonds}\n"
        f"saveamberparm COM_OUT {outdir / 'COM.prmtop'} {outdir / 'COM.inpcrd'}\n"
        "quit\n"
    )


def run_mmpbsa_py(
    mmpbsa_py: Path,
    input_file: Path,
    complex_prmtop: Path,
    receptor_prmtop: Path,
    ligand_prmtop: Path,
    trajectory: Path,
    output_dat: Path,
    output_csv: Path,
    root: Path,
    env: dict[str, str],
) -> None:
    run_command(
        [
            str(mmpbsa_py),
            "-O",
            "-i",
            input_file,
            "-cp",
            complex_prmtop,
            "-rp",
            receptor_prmtop,
            "-lp",
            ligand_prmtop,
            "-y",
            trajectory,
            "-o",
            output_dat,
            "-eo",
            output_csv,
        ],
        cwd=root,
        env=env,
    )


def run_binding_energy(
    root: Path,
    bound_gro: Path,
    bound_xtc: Path,
    outdir: Path,
    ligand_resname: str,
    ligand_charge: int,
    startframe: int,
    endframe: int,
    interval: int,
    run_gbsa: bool,
    run_pbsa: bool,
    pbsa_startframe: int,
    pbsa_endframe: int,
    pbsa_interval: int,
    gmx_exe: str,
    amber_bin: Path,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    gmx = Path(gmx_exe)
    gmx_mmpbsa = amber_bin / "gmx_MMPBSA"
    mmpbsa_py = amber_bin / "MMPBSA.py"
    antechamber = amber_bin / "antechamber"
    parmchk2 = amber_bin / "parmchk2"
    tleap = amber_bin / "tleap"
    env = tool_env(amber_bin, gmx)

    complex_pdb = outdir / "complex_view.pdb"
    default_ndx = outdir / "complex_default.ndx"
    gmx_input = outdir / "gmx_mmpbsa_quick.in"
    amber_input = outdir / "amber_mmpbsa_quick.in"
    pbsa_input = outdir / "amber_mmpbsa_pbsa.in"

    run_command([str(gmx), "editconf", "-f", bound_gro, "-o", complex_pdb], cwd=root, env=env)
    run_command([str(gmx), "make_ndx", "-f", bound_gro, "-o", default_ndx], cwd=root, input_text="q\n", env=env)

    groups = parse_ndx_groups(default_ndx)
    if "Protein" not in groups:
        raise RuntimeError(f"No Protein group found in {default_ndx}")
    if ligand_resname not in groups:
        raise RuntimeError(f"No ligand group named {ligand_resname} found in {default_ndx}")

    write_gbsa_input(gmx_input, startframe, endframe, interval, gmx_style=True)
    prep = run_command(
        [
            str(gmx_mmpbsa),
            "-O",
            "-i",
            gmx_input,
            "-cs",
            complex_pdb,
            "-ci",
            default_ndx,
            "-cg",
            str(groups["Protein"]),
            str(groups[ligand_resname]),
            "-ct",
            bound_xtc,
            "-o",
            outdir / "gmx_mmpbsa_probe.dat",
            "-eo",
            outdir / "gmx_mmpbsa_probe.csv",
            "-nogui",
        ],
        cwd=outdir,
        env=env,
        check=False,
    )
    if prep.returncode != 0:
        print("[INFO] gmx_MMPBSA setup did not finish; continuing with generated structure files.")

    rec_pdb = outdir / "_GMXMMPBSA_REC_F1.pdb"
    lig_pdb = outdir / "_GMXMMPBSA_LIG.pdb"
    if not rec_pdb.exists() or not lig_pdb.exists():
        raise RuntimeError("gmx_MMPBSA did not produce the receptor/ligand PDB files needed for fallback GBSA.")

    ligand_mol2 = outdir / f"{ligand_resname.lower()}_gaff.mol2"
    ligand_frcmod = outdir / f"{ligand_resname.lower()}_gaff.frcmod"
    run_command(
        [
            str(antechamber),
            "-i",
            lig_pdb,
            "-fi",
            "pdb",
            "-o",
            ligand_mol2,
            "-fo",
            "mol2",
            "-c",
            "bcc",
            "-s",
            "2",
            "-nc",
            str(ligand_charge),
            "-rn",
            ligand_resname,
        ],
        cwd=outdir,
        env=env,
    )
    run_command([str(parmchk2), "-i", ligand_mol2, "-f", "mol2", "-o", ligand_frcmod], cwd=outdir, env=env)

    disulfides = detect_disulfides(rec_pdb)
    print(f"[INFO] Disulfides detected: {disulfides}")
    leap_in = outdir / "manual_leap.in"
    write_manual_leap(leap_in, rec_pdb, ligand_mol2, ligand_frcmod, disulfides, outdir)
    run_command([str(tleap), "-f", leap_in], cwd=outdir, env=env)

    universe = mda.Universe(str(bound_gro))
    complex_atoms = universe.select_atoms(f"protein or resname {ligand_resname}")
    if len(complex_atoms) == 0:
        raise RuntimeError(f"Selection produced no atoms: protein or resname {ligand_resname}")
    strip_ndx = outdir / "complex_protein_ligand.ndx"
    write_single_group_ndx(complex_atoms.indices + 1, "Complex", strip_ndx)

    stripped_xtc = outdir / "complex_protein_ligand.xtc"
    run_command(
        [str(gmx), "trjconv", "-f", bound_xtc, "-s", complex_pdb, "-n", strip_ndx, "-o", stripped_xtc],
        cwd=root,
        input_text="0\n",
        env=env,
    )

    if run_gbsa:
        write_gbsa_input(amber_input, startframe, endframe, interval, gmx_style=False)
        run_mmpbsa_py(
            mmpbsa_py,
            amber_input,
            outdir / "COM.prmtop",
            outdir / "REC.prmtop",
            outdir / "LIG.prmtop",
            stripped_xtc,
            outdir / "quick_gbsa.dat",
            outdir / "quick_gbsa.csv",
            root,
            env,
        )
        print(f"[OK] GBSA outputs written to {outdir}")

    if run_pbsa:
        write_pbsa_input(pbsa_input, pbsa_startframe, pbsa_endframe, pbsa_interval)
        run_mmpbsa_py(
            mmpbsa_py,
            pbsa_input,
            outdir / "COM.prmtop",
            outdir / "REC.prmtop",
            outdir / "LIG.prmtop",
            stripped_xtc,
            outdir / "last20ns_pbsa.dat",
            outdir / "last20ns_pbsa.csv",
            root,
            env,
        )
        print(f"[OK] PBSA outputs written to {outdir}")


def default_endframe(xtc: Path, gro: Path) -> int:
    return len(mda.Universe(str(gro), str(xtc)).trajectory)


def startframe_for_last_ns(gro: Path, xtc: Path, last_ns: float) -> int:
    universe = mda.Universe(str(gro), str(xtc))
    start_ps = max(universe.trajectory[0].time, universe.trajectory[-1].time - last_ns * 1000.0)
    return frame_index_at_or_after(gro, xtc, start_ps) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RMSF comparison and optional quick GBSA/PBSA for packaged MD trajectories.")
    parser.add_argument("--root", default=".", help="Project root containing the trajectory paths.")
    parser.add_argument("--apo-gro", default="apo_monomer/apo_monomer_view.gro")
    parser.add_argument("--apo-xtc", default="apo_monomer/apo_monomer_view.xtc")
    parser.add_argument("--bound-gro", default="monomer_ligand_rep1/monomer_ligand_rep1_view.gro")
    parser.add_argument("--bound-xtc", default="monomer_ligand_rep1/monomer_ligand_rep1_view.xtc")
    parser.add_argument("--apo-label", default="Apo monomer")
    parser.add_argument("--bound-label", default="Ligand-bound monomer")
    parser.add_argument("--outdir", default="analysis/automated_outputs")
    parser.add_argument("--last-ns", type=float, default=10.0)
    parser.add_argument("--run-gbsa", action="store_true", help="Run the quick GBSA protein-ligand analysis.")
    parser.add_argument("--run-pbsa", action="store_true", help="Run the quick PBSA protein-ligand analysis.")
    parser.add_argument("--ligand-resname", default="UNL")
    parser.add_argument("--ligand-charge", type=int, default=1)
    parser.add_argument("--gbsa-startframe", type=int, default=None, help="1-based first frame for GBSA. Default: last 10 ns.")
    parser.add_argument("--gbsa-endframe", type=int, default=None, help="1-based final frame for GBSA. Default: final frame.")
    parser.add_argument("--gbsa-interval", type=int, default=5)
    parser.add_argument("--pbsa-last-ns", type=float, default=20.0, help="Last N ns used for PBSA if --pbsa-startframe is omitted.")
    parser.add_argument("--pbsa-startframe", type=int, default=None, help="1-based first frame for PBSA. Default: last --pbsa-last-ns.")
    parser.add_argument("--pbsa-endframe", type=int, default=None, help="1-based final frame for PBSA. Default: final frame.")
    parser.add_argument("--pbsa-interval", type=int, default=5)
    parser.add_argument(
        "--gmx-exe",
        default="/Users/ysb/miniforge3/envs/gmxmmpbsa/bin.ARM_NEON_ASIMD/gmx",
        help="GROMACS executable. Use the real executable, not 'conda run gmx', for stdin prompts.",
    )
    parser.add_argument("--amber-bin", default="/Users/ysb/miniforge3/envs/gmxmmpbsa/bin")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    outdir = (root / args.outdir).resolve()
    rmsf_out = outdir / "rmsf"
    gbsa_out = outdir / "gbsa"

    apo = SystemSpec(args.apo_label, root / args.apo_gro, root / args.apo_xtc, "#2166ac")
    bound = SystemSpec(args.bound_label, root / args.bound_gro, root / args.bound_xtc, "#b2182b")
    run_rmsf([apo, bound], rmsf_out, args.last_ns)

    if args.run_gbsa or args.run_pbsa:
        endframe = args.gbsa_endframe or default_endframe(bound.xtc, bound.gro)
        if args.gbsa_startframe is None:
            startframe = startframe_for_last_ns(bound.gro, bound.xtc, args.last_ns)
        else:
            startframe = args.gbsa_startframe

        pbsa_endframe = args.pbsa_endframe or default_endframe(bound.xtc, bound.gro)
        if args.pbsa_startframe is None:
            pbsa_startframe = startframe_for_last_ns(bound.gro, bound.xtc, args.pbsa_last_ns)
        else:
            pbsa_startframe = args.pbsa_startframe

        run_binding_energy(
            root=root,
            bound_gro=bound.gro,
            bound_xtc=bound.xtc,
            outdir=gbsa_out,
            ligand_resname=args.ligand_resname,
            ligand_charge=args.ligand_charge,
            startframe=startframe,
            endframe=endframe,
            interval=args.gbsa_interval,
            run_gbsa=args.run_gbsa,
            run_pbsa=args.run_pbsa,
            pbsa_startframe=pbsa_startframe,
            pbsa_endframe=pbsa_endframe,
            pbsa_interval=args.pbsa_interval,
            gmx_exe=args.gmx_exe,
            amber_bin=Path(args.amber_bin).expanduser().resolve(),
        )


if __name__ == "__main__":
    main()
