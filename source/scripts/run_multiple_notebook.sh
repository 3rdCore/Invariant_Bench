#!/bin/bash

script="contrastive"
echo "Running script: ${script}.ipynb"

pairs=(
  "0.5 0.01"
  "0.5 0.05"
  "0.5 0.1"
  "0.5 0.2"
  "0.5 0.5"
)

result_timestamp=$(date +%Y%m%d-%H%M%S)
# Print job submission information
echo "Results will be saved under results/${result_timestamp}/"

for seed in {0..4}
do
  for pair in "${pairs[@]}"; do
    attr_prob=$(echo $pair | awk '{print $1}')
    spur_prob=$(echo $pair | awk '{print $2}')
  folder=${result_timestamp}/${attr_prob}_${spur_prob}/${seed}
    
    # Create a temporary sbatch script with a unique name
    job_script="run_${script}_job_${attr_prob}_${spur_prob}_${seed}.sh"
    cat <<EOF > $job_script
#!/bin/bash
#SBATCH --job-name=${script}_${attr_prob}_${spur_prob}_${seed}
#SBATCH --output=../results/${folder}/${script}-%j.out
#SBATCH --error=../results/${folder}/${script}-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=long

module load cuda/11.8
cd /home/mila/t/tom.marty/invariant_bench/source
source ../.venv/bin/activate
export CMNIST_ATTR_PROB=$attr_prob
export CMNIST_SPUR_PROB=$spur_prob
export SEED=$seed
export RESULT_FOLDER=$folder
mkdir -p ../results/${RESULT_FOLDER}
jupyter nbconvert --to notebook --execute ${script}.ipynb --output ../results/${RESULT_FOLDER}/executed_notebook.ipynb
EOF

    # Submit the job
    sbatch $job_script
    rm -f $job_script
  done
done

echo "Notebook jobs submitted!"
