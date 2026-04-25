# ============================================================
#  CodeReviewEnv v3 — GRPO Training Notebook
#  Run this on Google Colab (T4/A100) or HuggingFace (A10G)
# ============================================================
#
# HOW TO USE:
# 1. Upload this entire Meta-project folder to Colab/HF
# 2. Select GPU runtime (A10G recommended, T4 works too)
# 3. Run all cells — training takes ~2-4 hours
# 4. Download grpo_output/ when done
# ============================================================

# ── Cell 1: Install dependencies ────────────────────────────
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps "trl>=0.17" peft accelerate bitsandbytes
# !pip install openenv-core fastmcp datasets matplotlib

# ── Cell 2: Verify GPU ──────────────────────────────────────
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU!'}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB" if torch.cuda.is_available() else "")
assert torch.cuda.is_available(), "ERROR: No GPU found! Switch to GPU runtime."

# ── Cell 3: Verify environment ──────────────────────────────
import sys, os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.getcwd()))

from code_review_env.server.environment import CodeReviewEnvironment
from openenv.core.env_server import CallToolAction, ListToolsAction

env = CodeReviewEnvironment()
obs = env.reset(difficulty="easy")
print(f"✅ Environment loaded: {len(obs.metadata['context'])} chars")
obs = env.step(ListToolsAction())
print(f"✅ Tools: {[t.name for t in obs.tools]}")

# ── Cell 4: Quick sanity check ──────────────────────────────
import re
ctx = env.reset(seed=42).metadata['context']
files = re.findall(r'• (.+?)\s+\[', ctx)
print(f"✅ Episode has {len(files)} files")

obs = env.step(CallToolAction(tool_name='read_file', arguments={'file_path': files[0]}))
print(f"✅ read_file works")

obs = env.step(CallToolAction(tool_name='flag_vulnerable', arguments={
    'file_path': files[0],
    'reasoning': 'Contains buffer overflow pattern in memcpy without bounds checking'
}))
print(f"✅ flag_vulnerable works")

obs = env.step(CallToolAction(tool_name='submit_report', arguments={
    'summary': 'Test report', 'confidence': 'low'
}))
result = str(obs.result.data if hasattr(obs.result, 'data') else obs.result)
score = re.search(r'TOTAL SCORE: ([\d.]+)', result)
print(f"✅ submit_report works — score: {score.group(1) if score else '?'}")
print(f"\n🎯 All checks passed! Ready to train.")

# ── Cell 5: TRAIN ───────────────────────────────────────────
# This is the main training cell. It will:
# 1. Load Qwen3-1.7B (thinking-mode) with 4-bit quantization (~1.5GB VRAM)
# 2. Apply LoRA adapters (r=16)
# 3. Generate 200 training prompts from the environment
# 4. Run GRPO training (2 epochs → ~400 optimizer steps)
# 5. Save the model and training curves

print("=" * 60)
print("  Starting GRPO Training...")
print("  This will take 2-4 hours on A10G, 4-6 hours on T4")
print("=" * 60)

# Run the training script
exec(open("train_grpo.py").read())

# ── Cell 6: View results ────────────────────────────────────
from IPython.display import Image, display
import json

# Show training curves
if os.path.exists("grpo_output/training_curves.png"):
    display(Image("grpo_output/training_curves.png"))
else:
    print("No training curves found yet.")

# Show stats
if os.path.exists("grpo_output/training_stats.json"):
    with open("grpo_output/training_stats.json") as f:
        stats = json.load(f)
    print(f"\n📊 Training Stats:")
    print(f"   Model: {stats['model']}")
    print(f"   Episodes: {stats['num_episodes']}")
    print(f"   Early mean score: {stats['early_mean']:.3f}")
    print(f"   Late mean score:  {stats['late_mean']:.3f}")
    print(f"   Improvement:      {stats['late_mean'] - stats['early_mean']:+.3f}")
    print(f"   Max score:        {stats['max_reward']:.3f}")

# ── Cell 7: Download model ──────────────────────────────────
# On Colab, use this to download:
# from google.colab import files
# !zip -r grpo_output.zip grpo_output/
# files.download("grpo_output.zip")

# On HuggingFace, the output is already in the Space.
print("\n✅ Training complete! Download grpo_output/ for your submission.")
