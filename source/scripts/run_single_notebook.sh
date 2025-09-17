#!/bin/bash

#SBATCH --job-name=pcl_single_run
#SBATCH --output=../results/${RESULT_FOLDER}/pcl-%j.out
#SBATCH --error=../results/${RESULT_FOLDER}/pcl-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=long

# Timestamp and folder setup
result_timestamp=$(date +%Y%m%d-%H%M%S)
RESULT_FOLDER=${result_timestamp}
export RESULT_FOLDER=$RESULT_FOLDER

# Load necessary modules
module load cuda/11.8

# Change to project directory
cd /home/mila/t/tom.marty/invariant_bench/source

source ../.venv/bin/activate

# Optionally set environment variables for single run
export CMNIST_ATTR_PROB=0.5
export CMNIST_SPUR_PROB=0.1
export SEED=0

# Create results folder
mkdir -p ../results/${RESULT_FOLDER}

# Run the notebook conversion command
jupyter nbconvert --to notebook --execute pcl.ipynb --output ../results/${RESULT_FOLDER}/executed_notebook.ipynb