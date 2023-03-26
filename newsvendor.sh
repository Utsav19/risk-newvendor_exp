#!/bin/bash
#SBATCH --job-name=news_compare_models
#SBATCH --account=def-edelage
#SBATCH --mail-user=utsav.sadana@mail.mcgill.ca
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=8
#SBATCH --output=chck.txt
#SBATCH --time=00-12:00
srun python main_risk.py
