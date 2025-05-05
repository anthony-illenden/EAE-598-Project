#!/bin/bash
#SBATCH --job-name="rf_model_training"
#SBATCH --output="rf_model_training.%j.%N.out.txt"
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --export=ALL
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH -t 04:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${SLURM_MAIL_USER}
#######################################################
source activate thesis
srun python rf_model_testing.py