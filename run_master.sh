#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p ml
#SBATCH -q ml
#SBATCH -A ml
#SBATCH -w concerto3
#SBATCH -c 32
#SBATCH --gpus-per-node=2
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --job-name=master_job
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err

# Initialize conda for this non-interactive shell
source /pkgs/anaconda3/etc/profile.d/conda.sh
conda activate /mfs1/u/chuning/jax_nqm

python master.py #test_job.py







