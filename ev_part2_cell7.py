# ==========================================
# CELL 7: LONG CONTEXT RECALL EVALUATION (ALL MODELS)
# ==========================================

evaluate_part2 = True
if evaluate_part2:
    import json, os
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from config import GPTConfig
    from tokenizer import BPETokenizer
    from model import GPT as AbsoluteGPT
    from model_relative import GPT as RelativeGPT

    tokenizer = BPETokenizer()
    tokenizer.load('merges.json')

    output_dir = "output"

    # --- 1. LOAD DATA ---
    print("📂 Loading evaluation dataset...")
    with open("long_context_recall_stories.jsonl", "r") as f:
        data = [json.loads(line) for line in f]

    distance_groups = {}
    for item in data:
        distance_groups.setdefault(item["distance"], []).append(item)
    sorted_distances = sorted(distance_groups.keys())

    # --- 2. HELPER ---
    def eval_model(model, is_absolute, block_size):
        losses = {}
        model.eval()
        with torch.no_grad():
            for dist in tqdm(sorted_distances, desc="   distances"):
                total = 0.0
                for ex in distance_groups[dist]:
                    input_ids = torch.tensor([tokenizer.encode(ex["text"])], dtype=torch.long, device=device)
                    if is_absolute:
                        input_ids = input_ids[:, -block_size:]
                    target = torch.tensor([ex["target_token_id"]], dtype=torch.long, device=device)
                    logits, _ = model(input_ids)
                    total += F.cross_entropy(logits[0, -1, :].unsqueeze(0), target).item()
                losses[dist] = total / len(distance_groups[dist])
        return losses

    # --- 3. COLLECT ALL RESULTS ---
    all_results = {}   # { informative_name: {dist: loss} }

    # Absolute baseline
    print("\n📥 Evaluating: Absolute_PE_Baseline")
    abs_config = GPTConfig.from_toml('config.toml')
    abs_model = AbsoluteGPT(abs_config).to(device)
    abs_model.load_state_dict(torch.load(f"{output_dir}/Arch_1_Baseline_weights.pt", map_location=device))
    all_results["Absolute_PE_Baseline"] = eval_model(abs_model, is_absolute=True, block_size=abs_config.block_size)

    # All relative models — auto-detect config from checkpoint
    rel_model_files = sorted([f for f in os.listdir(output_dir) if f.startswith("Rel_") and f.endswith(".pt")])

    for model_file in rel_model_files:
        label = model_file.replace("_weights.pt", "")
        print(f"\n📥 Evaluating: {label}")

        ckpt = torch.load(os.path.join(output_dir, model_file), map_location=device)

        rel_config = GPTConfig.from_toml('config.toml')
        if 'transformer.wte.weight' in ckpt:
            rel_config.n_embd = ckpt['transformer.wte.weight'].shape[1]
        layers = {int(k.split('.')[2]) for k in ckpt if k.startswith('transformer.h.')}
        if layers:
            rel_config.n_layer = max(layers) + 1
        if 'transformer.h.0.attn.E_k.weight' in ckpt:
            e_k_shape = ckpt['transformer.h.0.attn.E_k.weight'].shape
            rel_config.k = (e_k_shape[0] - 1) // 2
            rel_config.n_head = rel_config.n_embd // e_k_shape[1]

        print(f"   ⚙️  k={rel_config.k} | layers={rel_config.n_layer} | heads={rel_config.n_head} | embd={rel_config.n_embd}")

        rel_model = RelativeGPT(rel_config).to(device)
        rel_model.load_state_dict(ckpt)
        all_results[label] = eval_model(rel_model, is_absolute=False, block_size=abs_config.block_size)

    # --- 4. SAVE JSON ---
    # Convert int keys to str for JSON compatibility
    json_results = {
        name: {str(d): v for d, v in losses.items()}
        for name, losses in all_results.items()
    }
    json_path = f"{output_dir}/all_models_recall_losses.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=4)
    print(f"\n💾 Stats saved → {json_path}")

    # --- 5. PLOT ---
    plt.figure(figsize=(13, 6))
    for name, losses in all_results.items():
        dists = sorted(losses.keys())
        vals  = [losses[d] for d in dists]
        style = dict(marker='o', linestyle='--') if name == "Absolute_PE_Baseline" else dict(marker='s')
        plt.plot(dists, vals, label=name, **style)

    plt.axvline(x=abs_config.block_size, color='gray', linestyle=':', alpha=0.6, label='block_size=512')
    plt.xlabel("Distance (tokens between fact and answer)")
    plt.ylabel("Loss on correct answer token")
    plt.title("Long Context Recall: All Models Comparison")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = f"{output_dir}/all_models_recall_plot.png"
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"📊 Plot saved → {plot_path}")

else:
    print("⚠️ Skipping evaluation (evaluate_part2=False).")