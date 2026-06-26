#!/bin/bash
#SBATCH --job-name=story_generation
#SBATCH --output=outputs/gen_%j.out
#SBATCH --error=outputs/gen_%j.err
#SBATCH --partition=course
#SBATCH --qos=course
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=raveco@post.bgu.ac.il

# Initialize conda inside execution context
source $(conda info --base)/etc/profile.d/conda.sh
conda activate neuro_dl

# Create output folder structures if missing
mkdir -p outputs output

echo "🚀 Launching GPU story generation process..."
python generate_stories.py