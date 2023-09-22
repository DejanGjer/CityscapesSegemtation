#!/bin/bash
#SBATCH --job-name=seg_city
#SBATCH --ntasks=1
#SBATCH --nodelist=n17
#SBATCH --partition=cuda
#SBATCH --output slurm.%J.out
#SBATCH --error slurm.%J.err
#SBATCH --time=48:00:00

python cache_dataset.py