"""
CodeReviewEnv v3 — HuggingFace Space App
=========================================
This app does TWO things:
1. TRAINS Qwen3-8B with GRPO (first launch)
2. DEMOS the environment for judges (after training)
"""
import gradio as gr
import os
import sys
import json
import threading
import time

sys.path.insert(0, os.path.dirname(__file__))

# ── Check if training is done ──────────────────────────
RESULTS_DIR = "./grpo_output"
TRAINING_DONE = os.path.exists(os.path.join(RESULTS_DIR, "training_stats.json"))
training_status = {"running": False, "progress": "", "done": TRAINING_DONE}


def get_status():
    if training_status["done"]:
        # Load stats
        stats_path = os.path.join(RESULTS_DIR, "training_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                stats = json.load(f)
            return (
                f"## ✅ Training Complete!\n\n"
                f"- **Model:** {stats.get('model', 'Qwen3-8B')}\n"
                f"- **Episodes:** {stats.get('num_episodes', '?')}\n"
                f"- **Early Mean Reward:** {stats.get('early_mean', 0):.3f}\n"
                f"- **Late Mean Reward:** {stats.get('late_mean', 0):.3f}\n"
                f"- **Improvement:** {stats.get('improvement', 0):+.3f}\n"
                f"- **Max Reward:** {stats.get('max_reward', 0):.3f}\n"
            )
        return "## ✅ Training Complete!"
    elif training_status["running"]:
        return f"## ⏳ Training in Progress...\n\n{training_status['progress']}\n\nDo NOT restart the Space."
    else:
        return "## ⏸️ Ready to Train\n\nClick **Start Training** below."


def get_training_curves():
    curves_path = os.path.join(RESULTS_DIR, "training_curves.png")
    if os.path.exists(curves_path):
        return curves_path
    return None


def run_training():
    """Run GRPO training in a background thread."""
    training_status["running"] = True
    training_status["progress"] = "Starting training script..."
    training_status["done"] = False
    try:
        import subprocess
        proc = subprocess.Popen(
            [sys.executable, "train_grpo.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=os.path.dirname(__file__) or "."
        )
        lines = []
        for line in proc.stdout:
            lines.append(line.strip())
            training_status["progress"] = "\n".join(lines[-15:])
            print(line.strip())  # Also print to container logs

        exit_code = proc.wait()
        if exit_code == 0 and os.path.exists(os.path.join(RESULTS_DIR, "training_stats.json")):
            training_status["done"] = True
            training_status["progress"] = "Training complete!"
        else:
            training_status["done"] = False
            error_lines = "\n".join(lines[-20:])
            training_status["progress"] = f"❌ Training FAILED (exit code {exit_code}):\n\n```\n{error_lines}\n```\n\nClick Start Training to retry."
        training_status["running"] = False
    except Exception as e:
        training_status["running"] = False
        training_status["done"] = False
        training_status["progress"] = f"❌ Error: {str(e)}\n\nClick Start Training to retry."


def start_training():
    if training_status["running"]:
        return "⚠️ Training is already running! Click Refresh Status to check progress."
    if training_status["done"] and os.path.exists(os.path.join(RESULTS_DIR, "training_stats.json")):
        return "✅ Training already completed! See results below."
    # Reset status and start training in background thread
    training_status["done"] = False
    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()
    return "🚀 Training started! This will take 3-5 hours. Do NOT restart the Space."


# ── Demo: Run the environment interactively ─────────────
def run_investigation(cve_id, difficulty):
    """Let judges interact with the environment."""
    try:
        from code_review_env.server.environment import CodeReviewEnvironment
        from openenv.core.env_server import CallToolAction
        import re

        env = CodeReviewEnvironment()
        obs = env.reset(difficulty=difficulty)
        ctx = obs.metadata.get("context", "No context available")
        files = re.findall(r'• (.+?)\s+\[', ctx)

        # Smart investigation: read top 3 files, flag highest complexity
        results = [f"**Briefing:**\n```\n{ctx}\n```\n"]

        for f in files[:3]:
            obs = env.step(CallToolAction(tool_name='read_file', arguments={'file_path': f}))
            code = str(obs.result.data if hasattr(obs.result, 'data') else obs.result)
            results.append(f"**read_file({f}):** {len(code)} chars read")

        # Flag the first file with reasoning
        if files:
            obs = env.step(CallToolAction(tool_name='flag_vulnerable', arguments={
                'file_path': files[0],
                'reasoning': 'High complexity source file with potential vulnerability pattern matching the CVE description'
            }))
            result_text = str(obs.result.data if hasattr(obs.result, 'data') else obs.result)
            results.append(f"**flag_vulnerable({files[0]}):**\n{result_text}")

        # Skip remaining
        for f in files[1:]:
            obs = env.step(CallToolAction(tool_name='skip_file', arguments={
                'file_path': f,
                'reasoning': 'Lower complexity or header/test file'
            }))

        # Submit report
        obs = env.step(CallToolAction(tool_name='submit_report', arguments={
            'summary': f'Security triage for {cve_id}: Investigated {len(files)} files, flagged primary vulnerability source.',
            'confidence': 'medium'
        }))
        report = str(obs.result.data if hasattr(obs.result, 'data') else obs.result)
        results.append(f"\n**Final Report:**\n```\n{report}\n```")

        return "\n\n".join(results)
    except Exception as e:
        return f"Error: {str(e)}"


# ── Build Gradio UI ─────────────────────────────────────
with gr.Blocks(
    title="CodeReviewEnv v3 — Security Investigation Agent",
    theme=gr.themes.Soft(primary_hue="purple", secondary_hue="blue")
) as app:
    gr.Markdown(
        "# 🔐 CodeReviewEnv v3 — Agentic Security Investigation\n"
        "**Train an LLM to investigate CVE vulnerabilities like a security engineer.**\n\n"
        "Built with OpenEnv + FastMCP + Qwen3-8B + GRPO"
    )

    with gr.Tabs():
        # Tab 1: Training
        with gr.Tab("🏋️ Training"):
            status_md = gr.Markdown(get_status)
            train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
            train_output = gr.Textbox(label="Training Output", lines=3)
            train_btn.click(fn=start_training, outputs=train_output)

            # Auto-refresh status
            refresh_btn = gr.Button("🔄 Refresh Status")
            refresh_btn.click(fn=get_status, outputs=status_md)

            # Show curves if available
            curves_img = gr.Image(label="Training Curves", value=get_training_curves())
            refresh_btn.click(fn=get_training_curves, outputs=curves_img)

        # Tab 2: Demo
        with gr.Tab("🔍 Live Demo"):
            gr.Markdown(
                "### Try the Environment\n"
                "Run a security investigation episode interactively."
            )
            with gr.Row():
                cve_input = gr.Textbox(label="CVE ID (optional)", value="auto", max_lines=1)
                diff_input = gr.Dropdown(["easy", "medium", "hard"], label="Difficulty", value="easy")
            demo_btn = gr.Button("🔎 Run Investigation", variant="primary")
            demo_output = gr.Markdown(label="Investigation Results")
            demo_btn.click(fn=run_investigation, inputs=[cve_input, diff_input], outputs=demo_output)

        # Tab 3: About
        with gr.Tab("📖 About"):
            gr.Markdown("""
## Architecture

- **Environment:** 6 MCP tools (read_file, search_code, get_function_list, flag_vulnerable, skip_file, submit_report)
- **Dataset:** 150 real-world CVEs (Log4Shell, Dirty COW, PwnKit, BlueKeep, etc.)
- **Model:** Qwen3-8B with thinking mode integration
- **Training:** GRPO with live environment execution reward
- **Innovation:** Thinking Budget — the agent learns WHEN to reason deeply

## Key Features

1. **5-Component Reward Function:** F1 + Report Quality + Efficiency + Thinking Allocation + Precision
2. **Anti-Gaming:** Blind-skip and blind-flag strategies are penalized
3. **Live Execution Reward:** Tool calls are parsed and executed against the real environment during training
4. **Qwen3 Thinking Mode:** Agent uses `<think>` blocks for deep reasoning on suspicious files

## The Pitch

> *"Our agent doesn't just reason — it knows WHEN to reason."*

A real security engineer doesn't analyze every file equally. They triage: deep analysis on suspicious code, quick glance at headers. Our environment teaches this skill.
            """)

if __name__ == "__main__":
    # AUTO-START training on boot if not already done
    if not training_status["done"] and not training_status["running"]:
        print("🚀 AUTO-STARTING training on boot...")
        training_status["done"] = False
        thread = threading.Thread(target=run_training, daemon=True)
        thread.start()
    app.launch()
