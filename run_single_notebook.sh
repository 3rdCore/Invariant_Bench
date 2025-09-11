
#!/bin/bash
#SBATCH --job-name=pcl_notebook
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=main

# Folder management logic
result_timestamp=$(date +%Y%m%d-%H%M%S)
RESULT_FOLDER=${result_timestamp}/single_run
export RESULT_FOLDER=$RESULT_FOLDER

# Output and error files in results folder
#SBATCH --output=subpopbench/results/${RESULT_FOLDER}/pcl-%j.out
#SBATCH --error=subpopbench/results/${RESULT_FOLDER}/pcl-%j.err

# Load necessary modules (adjust based on your environment)
module load python/3.9
module load cuda/11.8

# Change to the project directory
cd /home/mila/t/tom.marty/invariant_bench/SubpopBench/subpopbench

source ../../XRM/.venv/bin/activate

# Create results folder
mkdir -p results/${RESULT_FOLDER}

# Run the notebook conversion command
jupyter nbconvert --to notebook --execute pcl.ipynb --output results/${RESULT_FOLDER}/executed_notebook.ipynb

echo "Notebook execution completed! Results saved in results/${RESULT_FOLDER}/"
