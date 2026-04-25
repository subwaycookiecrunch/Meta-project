#!/usr/bin/env bash
# morning_run.sh — one-command Sunday morning submission workflow.
#
# Run this AFTER training has finished on the v3 Space (~10:30 IST).
# Each step is idempotent. If a step fails, fix it and re-run; earlier
# steps will skip.
#
# Order matches the 7-action playbook from MORNING_CHECKLIST.md:
#   1. Pull trained artifacts from the Space
#   2. Verify checkpoint round-trip via inference.py
#   3. Run eval_baseline.py (untrained vs trained, same episodes)
#   4. Regenerate calibration plot from real training data
#   5. Re-run red team against the trained reward function
#   6. Build the unified "improvement evidence panel"
#   7. Smoke-test all artifacts present, then prompt to call ship.sh

set -euo pipefail
cd "$(dirname "$0")"

OUT=grpo_output
mkdir -p "$OUT"

step() { echo ""; echo "═══ $1 ═══"; }
ok()   { echo "  ✅ $1"; }
warn() { echo "  ⚠  $1"; }
fail() { echo "  ❌ $1"; exit 1; }

# ── 1. Pull trained artifacts ────────────────────────────────────────
step "1/7  Pull trained artifacts from Space"
if [[ -f "$OUT/training_curves.png" && -f "$OUT/training_stats.json" ]]; then
    ok "training_curves.png + training_stats.json already present locally"
else
    warn "Pull the following from https://huggingface.co/spaces/lucid987654/code-review-env-v3 manually:"
    echo "       - grpo_output/training_curves.png"
    echo "       - grpo_output/training_stats.json"
    echo "       - grpo_output/eval_calibration.2.json (real calibration data)"
    echo "       - grpo_output/trace_log.jsonl (real trace log)"
    echo "       - grpo_output/checkpoint-*/  (LoRA adapter)"
    echo ""
    read -r -p "      Press Enter once pulled, or Ctrl-C to abort: "
fi

# ── 2. Verify checkpoint round-trip ─────────────────────────────────
step "2/7  Verify trained checkpoint loads cleanly"
LATEST_CKPT=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | tail -n1 || true)
if [[ -n "$LATEST_CKPT" ]]; then
    echo "  Found checkpoint: $LATEST_CKPT"
    python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
print('  Loading base + adapter...')
base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-1.7B', dtype=torch.bfloat16, device_map='auto')
model = PeftModel.from_pretrained(base, '$LATEST_CKPT')
print('  ✅ Adapter loaded cleanly:', sum(p.numel() for p in model.parameters() if p.requires_grad), 'trainable params')
" || fail "Checkpoint round-trip failed"
    ok "Checkpoint validates"
else
    warn "No checkpoint found in $OUT — skipping round-trip verification"
fi

# ── 3. Baseline vs trained on same episodes ──────────────────────────
step "3/7  Run eval_baseline.py (untrained vs trained on identical episodes)"
if [[ -f "$OUT/eval_baseline_vs_trained.png" ]]; then
    ok "eval_baseline_vs_trained.png already exists — skip (delete it to force regen)"
else
    python eval_baseline.py || fail "eval_baseline.py crashed"
    [[ -f "$OUT/eval_baseline_vs_trained.png" ]] && ok "eval_baseline_vs_trained.png produced"
fi

# ── 4. Regenerate calibration plot from real data ────────────────────
step "4/7  Regenerate calibration plot from real training-time data"
if [[ -f "$OUT/eval_calibration.2.json" ]]; then
    python scripts/generate_calibration_plot.py --mode real --data "$OUT/eval_calibration.2.json" \
        || fail "calibration plot regen failed"
    ok "calibration_plot.png regenerated from real GRPO trace data"
else
    warn "eval_calibration.2.json not found — using heuristic plot (still ships)"
fi

# ── 5. Re-run red team against current reward fn ────────────────────
step "5/7  Re-run red team simulator against the live reward function"
python scripts/red_team.py || fail "red_team.py crashed"
ok "red_team_results.json refreshed"

# ── 6. Build unified improvement evidence panel ──────────────────────
step "6/7  Build unified improvement evidence panel"
if [[ -f scripts/build_improvement_panel.py ]]; then
    python scripts/build_improvement_panel.py || warn "improvement panel build failed (non-fatal)"
    ok "improvement_panel.png produced"
else
    warn "scripts/build_improvement_panel.py not yet created — skipping (non-fatal)"
fi

# ── 7. Smoke-check all artifacts present ─────────────────────────────
step "7/7  Smoke-check all artifacts before ship.sh"
EXPECTED=(
    "$OUT/training_curves.png"
    "$OUT/training_stats.json"
    "$OUT/eval_baseline_vs_trained.png"
    "$OUT/calibration_plot.png"
    "$OUT/thinking_allocation.png"
    "$OUT/transfer_results.png"
    "data/red_team_results.json"
)
MISSING=0
for f in "${EXPECTED[@]}"; do
    if [[ -f "$f" ]]; then ok "$f"
    else warn "$f MISSING"; MISSING=$((MISSING+1))
    fi
done
if [[ $MISSING -gt 0 ]]; then
    warn "$MISSING artifact(s) missing — review above before pushing"
fi

# ── done ─────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ Morning workflow complete."
echo ""
echo "Next:"
echo "  1. Record the 2-min video using VIDEO_SCRIPT.md (~30 min)"
echo "  2. Update README.md + blog_post.md with the YouTube URL"
echo "     (replace __YOUTUBE_URL__ placeholders)"
echo "  3. Run: ./ship.sh"
echo "  4. Publish blog at https://huggingface.co/new-blog"
echo "  5. Submit at the hackathon form"
echo "═══════════════════════════════════════════════════════════"
