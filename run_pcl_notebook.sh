#!/bin/bash
#SBATCH --job-name=pcl_notebook
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=main

# Load necessary modules (adjust based on your environment)
module load python/3.9
module load cuda/11.8

# Activate your conda environment if needed
# conda activate your_env_name

# Change to the project directory
cd /home/mila/t/tom.marty/invariant_bench/SubpopBench/

source ../XRM/.venv/bin/activate
# Run the notebook conversion command
jupyter nbconvert --to notebook --execute subpopbench/pcl.ipynb --output executed_notebook_1.ipynb

echo "Notebook execution completed!"
