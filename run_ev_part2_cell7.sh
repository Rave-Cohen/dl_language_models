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

# 3. Create folders
mkdir -p outputs
mkdir -p output 

# 4. Run the script (If you must use the notebook)
echo "🚀 Starting job on $(hostname)"

# It is highly recommended to run a .py file instead of a .ipynb file
# If you must run the notebook, use:
python ev_part2_cell7.py
echo "🎉 Job finished!"