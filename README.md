
# Invariant Feature Learning Benchmark

This project benchmarks invariant feature learning algorithms under varying spurious correlations and dataset sizes. Experiments and analysis are managed via Jupyter Notebooks.

## Quick Start

### 1. Installation
Clone the repo and set up the Python environment (skip `uv` steps on Mila cluster):
```bash
git clone https://github.com/3rdCore/Invariant_Bench.git
cd Invariant_Bench/
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Store Results on Scratch
To avoid filling your home directory, store results on scratch and symlink for compatibility:
```bash
mkdir -p ~/scratch/invariant_bench/results
ln -s ~/scratch/invariant_bench/results ~/invariant_bench/results
```

### 3. Download Data
Download datasets to scratch:
```bash
python scripts/download.py --download --data_path ~/scratch/data/benchmark cmnist
```

## Workflow

### Main Experiment Notebook
- `pcl.ipynb`: Trains algorithms (ERM, GroupDRO, etc.) on ColoredMNIST with controlled spurious correlations. Saves metrics to CSV.

### Analysis Notebook
- `results/analysis/analysis.ipynb`: Merges CSVs and generates plots for learning curves, algorithm comparison, and spurious correlation impact.

## Running Experiments

### Single Experiment
Run with default parameters:
```bash
sbatch scripts/run_single_notebook.sh
```
Results are saved in a timestamped folder in `results/`.

### Sweep of Experiments
Run multiple seeds and spurious settings:
```bash
./scripts/run_multiple_notebook.sh
```
Results are organized by parameter combination and seed.

## Analyzing Results

Open `results/analysis/analysis.ipynb` in Jupyter. Set `local_data_path` to your CSV results folder (e.g., `results/data`). Run all cells to merge CSVs and plot metrics.

## Acknowledgements
Based on open-source implementations from [source](https://github.com/YyzHarry/source), [DomainBed](https://github.com/facebookresearch/DomainBed), [spurious_feature_learning](https://github.com/izmailovpavel/spurious_feature_learning), and [multi-domain-imbalance](https://github.com/YyzHarry/multi-domain-imbalance).
