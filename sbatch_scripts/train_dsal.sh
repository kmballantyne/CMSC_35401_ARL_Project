#!/bin/bash
#SBATCH --job-name=dsal_training
#SBATCH --output=/home/kballantyne/CMSC_35401_ARL_Project/logs/%j.%N.stdout
#SBATCH --error=/home/kballantyne/CMSC_35401_ARL_Project/logs/%j.%N.stderr
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kballantyne@cs.uchicago.edu
#SBATCH --chdir=/home/kballantyne/CMSC_35401_ARL_Project
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00

# Load shell environment
source /home/${USER}/.bashrc
# Activate your conda environment
source activate ARL_Project

# Run your training script
python3 main.py
