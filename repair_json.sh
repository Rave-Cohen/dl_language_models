#!/bin/bash
#SBATCH --chdir=/home/raveco/deep_learning_ass3
#SBATCH --job-name=reconstruct_json
#SBATCH --output=outputs/reconstruct_%j.out
#SBATCH --error=outputs/reconstruct_%j.err
#SBATCH --partition=course
#SBATCH --qos=course
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=raveco@post.bgu.ac.il

source $(conda info --base)/etc/profile.d/conda.sh
conda activate neuro_dl
mkdir -p outputs output

echo "🚀 Reconstructing all_tuning_stats.json on $(hostname)"
python - <<'EOF'
import sys, os
sys.path.append('.')  # <--- ADD THIS LINE
import torch, json, os
from config import GPTConfig
from model import GPT
from torch.utils.data import DataLoader

device = 'cuda'
config = GPTConfig.from_toml('config.toml')
output_dir = 'output'

# Reuse pre-tokenized val set
class PreTokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, block_size):
        self.tokens, self.block_size = tokens, block_size
    def __len__(self): return len(self.tokens) - self.block_size
    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]

val_dataset = PreTokenizedDataset(torch.load('pretokenized_val_5k.pt'), config.block_size)
val_loader  = DataLoader(val_dataset, batch_size=32, shuffle=False)

def estimate_loss(model, loader, eval_iters=50):
    model.eval()
    losses = torch.zeros(eval_iters)
    with torch.no_grad():
        for i, (X, Y) in enumerate(loader):
            if i >= eval_iters: break
            _, loss = model(X.to(device), targets=Y.to(device))
            losses[i] = loss.item()
    return losses.mean().item()

# Trial 5 intentionally excluded (OOM)
trials = ["Trial_1_Baseline", "Trial_2_Fast_Learner", "Trial_3_High_Reg", "Trial_4_Conservative"]

stats = {}
for name in trials:
    pt = os.path.join(output_dir, f"{name}_best.pt")
    m = GPT(config).to(device)
    m.load_state_dict(torch.load(pt, map_location=device))
    val_loss = estimate_loss(m, val_loader)
    stats[name] = {"best_val_loss": val_loss, "history": []}
    print(f"  {name}: {val_loss:.4f}")

with open('/home/raveco/deep_learning_ass3/output/all_tuning_stats.json', 'w') as f:
    json.dump(stats, f, indent=4)
print("✅ Saved all_tuning_stats.json")
EOF

echo "🎉 Done!"