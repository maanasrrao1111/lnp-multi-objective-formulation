#!/bin/bash -l
#SBATCH --job-name=lnp_phase_c
#SBATCH --time=00:30:00
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=logs/phase_c_%j.out
#SBATCH --error=logs/phase_c_%j.err
#SBATCH --mail-type=end
#SBATCH --mail-user=YOUR_JHED@jhu.edu

# --- environment ---
module load anaconda
conda activate lnp-project

# --- run ---
cd $SLURM_SUBMIT_DIR
mkdir -p logs
python scripts/phase_c_feature_engineering.py
