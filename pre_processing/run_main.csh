#!/bin/bash
#SBATCH --job-name="run_main"
#SBATCH --output="run_main.%j.%N.out.txt"
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128
#SBATCH --export=ALL
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH -t 04:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${SLURM_MAIL_USER}
#######################################################
source activate thesis
srun python main.py