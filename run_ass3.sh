#!/bin/bash
#SBATCH --job-name=ass3_notebook
#SBATCH --output=outputs/ass3_train_%j.out
#SBATCH --error=outputs/ass3_train_%j.err
#SBATCH --partition=course
#SBATCH --qos=course
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=raveco@post.bgu.ac.il

# 1. Initialize Conda
source $(conda info --base)/etc/profile.d/conda.sh

# 2. Activate your environment
conda activate neuro_dl

# 3. Create necessary folders (Fixed paths)
mkdir -p outputs
mkdir -p output 

# 4. Run the notebook
echo "🚀 Starting Assignment 3 Notebook Execution on $(hostname)"

# Fixed path: Just look for main.ipynb in the current directory!
jupyter nbconvert --to notebook --execute --inplace main.ipynb

echo "🎉 Job finished!"