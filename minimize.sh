#!/bin/bash
set -e

# ----------------------------------------------------------
# Parse GPU flag
#Example usage: bash minimize.sh --gpu 0
# ----------------------------------------------------------
GPU_ID=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash minimize.sh --gpu [0|1|2|3]"
            exit 1
            ;;
    esac
done

echo "Using GPU: $GPU_ID"

# ----------------------------------------------------------
# Auto-detect CIF
# ----------------------------------------------------------
CIFS=( *.cif )

if (( ${#CIFS[@]} == 0 )); then
    echo "ERROR: No .cif file found in this directory."
    exit 1
fi

if (( ${#CIFS[@]} > 1 )); then
    echo "ERROR: Multiple .cif files found. Keep only one:"
    printf "%s\n" "${CIFS[@]}"
    exit 1
fi

CIF="${CIFS[0]}"
echo "Detected CIF: $CIF"

# ----------------------------------------------------------
# 1. CIF -> PDB
# ----------------------------------------------------------
obabel "$CIF" -O protein.pdb

# ----------------------------------------------------------
# 2. Topology (CHARMM36 + TIP3P)
# ----------------------------------------------------------
printf "9\n1\n0\n0\n" | gmx pdb2gmx \
    -f protein.pdb \
    -o protein.gro \
    -p topol.top \
    -i posre.itp \
    -ignh -ter

# ----------------------------------------------------------
# 3. Box
# ----------------------------------------------------------
gmx editconf -f protein.gro -o boxed.gro -c -d 1.0 -bt dodecahedron

# ----------------------------------------------------------
# 4. Solvate
# ----------------------------------------------------------
gmx solvate -cp boxed.gro -o solv.gro -p topol.top

# ----------------------------------------------------------
# 5. Mini ion EM .mdp
# ----------------------------------------------------------
cat > ions_em.mdp <<EOF
integrator      = steep
emtol           = 1000
nsteps          = 500
cutoff-scheme   = Verlet
coulombtype     = PME
rcoulomb        = 1.2
vdwtype         = cut-off
rvdw            = 1.2
constraints     = h-bonds
pbc             = xyz
EOF

gmx grompp -f ions_em.mdp -c solv.gro -p topol.top -o ions_NaCl.tpr -maxwarn 1

# ----------------------------------------------------------
# 6. Add NaCl
# ----------------------------------------------------------
printf "SOL\n" | gmx genion \
  -s ions_NaCl.tpr \
  -o solv_NaCl.gro \
  -p topol.top \
  -neutral -conc 0.15 \
  -pname NA -nname CL

# ----------------------------------------------------------
# 7. Single-step EM
# ----------------------------------------------------------
cat > em_single.mdp <<EOF
integrator      = steep
emtol           = 50
emstep          = 0.01
nsteps          = 50000
cutoff-scheme   = Verlet
coulombtype     = PME
rcoulomb        = 1.2
vdwtype         = cut-off
rvdw            = 1.2
constraints     = h-bonds
pbc             = xyz
DispCorr        = EnerPres
EOF

gmx grompp -f em_single.mdp \
    -c solv_NaCl.gro \
    -p topol.top \
    -o em_single.tpr

echo "Running EM on GPU $GPU_ID"

CUDA_VISIBLE_DEVICES=$GPU_ID gmx mdrun \
    -deffnm em_single \
    -ntmpi 1 \
    -ntomp 22

# ----------------------------------------------------------
# 8. Output
# ----------------------------------------------------------
gmx editconf -f em_single.gro -o minimized.pdb

echo "Done. Output: minimized.pdb"
