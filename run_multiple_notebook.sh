#!/bin/bash

# Change to the project directory
cd /home/mila/t/tom.marty/invariant_bench/SubpopBench/

for attr_prob in 0.01 0.1 0.5; do
  for spur_prob in 0.01 0.1 0.5; do
    export CMNIST_ATTR_PROB=$attr_prob
    export CMNIST_SPUR_PROB=$spur_prob
    export RESULTS_TIMESTAMP=$(date +%Y%m%d-%H%M%S)_${attr_prob}_${spur_prob}

    # Create a temporary sbatch script, launch it on the cluster, then delete it
    job_script="run_pcl_job_${RESULTS_TIMESTAMP}.sh"
    cat <<EOF > $job_script
#!/bin/bash
#SBATCH --job-name=pcl_${attr_prob}_${spur_prob}
#SBATCH --output=subpopbench/results/${RESULTS_TIMESTAMP}/pcl_${attr_prob}_${spur_prob}-%j.out
#SBATCH --error=subpopbench/results/${RESULTS_TIMESTAMP}/pcl_${attr_prob}_${spur_prob}-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=long

module load cuda/11.8
cd /home/mila/t/tom.marty/invariant_bench/SubpopBench/subpopbench
source ../../XRM/.venv/bin/activate
export CMNIST_ATTR_PROB=$attr_prob
export CMNIST_SPUR_PROB=$spur_prob
export RESULTS_TIMESTAMP=$RESULTS_TIMESTAMP
mkdir -p results/${RESULTS_TIMESTAMP}
jupyter nbconvert --to notebook --execute pcl.ipynb --output results/${RESULTS_TIMESTAMP}/executed_notebook_attr${attr_prob}_spur${spur_prob}.ipynb
EOF

    # Submit the job
    sbatch $job_script
    rm -f $job_script
  done

done

echo "Notebook jobs submitted!"
