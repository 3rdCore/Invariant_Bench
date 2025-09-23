#!/bin/bash
#SBATCH --job-name=${script}_single_run
#SBATCH --output=../results/${script}-%j.out
#SBATCH --error=../results/${script}-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=long

script="pcl"  # Options: "contrastive", "pcl", "analysis"
load_pretrained="False"  # Set to "True" to enable pretrained models

echo "Running script: ${script}.ipynb (hardcoded)"


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
export LOAD_PRETRAINED=$load_pretrained

# Create results folder
mkdir -p ../results/${RESULT_FOLDER}

# Run the notebook conversion command
jupyter nbconvert --to notebook --execute ${script}.ipynb --output-dir ../results/${RESULT_FOLDER} --output executed_${script}_notebook.ipynb