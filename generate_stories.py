import sys
import os
import json
import torch

# Ensure Python looks in the current working directory for local imports
sys.path.append('.')

from config import GPTConfig
from tokenizer import BPETokenizer
from model import GPT

# Set environment paths and configurations
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️  Using Device: {device}")

# Enforce deterministic tracking
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

config = GPTConfig.from_toml('config.toml')
assert config.block_size == 512, "ERROR: Teacher said DO NOT change block_size!"

tokenizer = BPETokenizer()
tokenizer.load('merges.json')
print("📚 Config and Tokenizer ready.")

# ---- Architecture Map Configuration ----
def _cfg(**kwargs):
    c = GPTConfig.from_toml('config.toml')
    for k, v in kwargs.items(): setattr(c, k, v)
    return c

arch_configs = {
    "Arch_1_Baseline":     _cfg(embd_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1),
    "Arch_2_Deep_Narrow":  _cfg(n_layer=8, n_head=4, n_embd=256, embd_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1),
    "Arch_3_Shallow_Wide": _cfg(n_layer=4, n_head=8, n_embd=512, embd_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1),
    "Arch_4_High_Dropout": _cfg(embd_pdrop=0.3, resid_pdrop=0.3, attn_pdrop=0.3),
}

# Auto-select the absolute best weights tracking from your training results json
stats_path = os.path.join(output_dir, 'architecture_comparison_stats.json')
if not os.path.exists(stats_path):
    print(f"❌ Error: Cannot find tracking file at {stats_path}!")
    sys.exit(1)

with open(stats_path, 'r') as f:
    data = json.load(f)

best_arch = min(data, key=lambda k: min(data[k]['val_loss']))
print(f"🏆 Best arch: {best_arch}  (best val_loss = {min(data[best_arch]['val_loss']):.4f})")

# Load best architecture
best_model = GPT(arch_configs[best_arch]).to(device)
weights_path = os.path.join(output_dir, f'{best_arch}_weights.pt')
best_model.load_state_dict(torch.load(weights_path, map_location=device))
best_model.eval()
print(f"✅ Loaded weights from {weights_path}\n")

# ---- Generation Engine with Tail Truncation ----
def generate_story(prompt, max_new_tokens=200, temperature=0.8, top_k=50):
    ids = tokenizer.encode(prompt)          
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = best_model.generate(idx, max_new_tokens=max_new_tokens,
                                  temperature=temperature, do_sample=True, top_k=top_k)
    
    # Clean output stream decoding
    story = tokenizer.decode(out[0].tolist()).rstrip()
    
    # Slice off hanging next sequence strings after the last period
    if '.' in story:
        story = story[:story.rfind('.') + 1]
        
    return story

# ---- Prompts Execution & Storage ----
prompts = [
    "Once upon a time",
    "In a small village",
    "There was a little dragon",
    "The sun was setting when",
    "A brave rabbit named Shuki",
]

# This dictionary handles storing the files cleanly
saved_stories = {
    "meta": {
        "selected_architecture": best_arch,
        "parameters": {"max_new_tokens": 200, "temperature": 0.8, "top_k": 50}
    },
    "stories": {}
}

for i, prompt in enumerate(prompts, 1):
    print(f"--- Story {i} ---")
    generated_text = generate_story(prompt)
    print(generated_text)
    print()
    
    # Store data with structural indexing keys
    saved_stories["stories"][f"story_{i}"] = {
        "prompt": prompt,
        "story": generated_text
    }

# Save output to output folder context
json_output_path = os.path.join(output_dir, 'generated_stories_results.json')
with open(json_output_path, 'w', encoding='utf-8') as f:
    json.dump(saved_stories, f, indent=4, ensure_ascii=False)

print(f"✅ All stories recorded and saved safely to: {json_output_path}")
print("🎉 Generation tasks completed successfully!")