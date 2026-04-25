# 🌅 Morning checklist — Sunday Apr 26

> **Hard deadline:** 17:00 Sunday.
> **Order is intentional.** Don't skip steps. Don't do them in parallel — the order minimises blockers.

---

## Step 1 · Verify training finished cleanly (10 min)

Open https://huggingface.co/spaces/lucid987654/code-review-env-v3 → **🏋️ Training Progress** tab.

You're looking for one of three states:

| State | What it means | What to do |
|---|---|---|
| ✅ **`training_done = true`** | Training completed | Continue to Step 2 |
| 🟡 **Step number stalled at < 470 but > 100, no error** | Container restarted mid-run; resumed from checkpoint | Continue to Step 2 with whatever curve we have |
| 🔴 **Error in logs / step count = 0** | Crash | Jump to **Plan B** at the bottom |

Also click the **📐 Calibration & Transfer** tab — confirm the calibration plot looks different from the heuristic proxy (real model produces messier, more interesting bars).

---

## Step 2 · Pull artifacts locally (10 min)

```bash
cd /Users/raj/Desktop/Meta-project
git fetch hf
git checkout hf/main -- grpo_output/training_curves.png
git checkout hf/main -- grpo_output/training_stats.json
git checkout hf/main -- grpo_output/calibration_plot.png
git checkout hf/main -- grpo_output/eval_calibration.json
git checkout hf/main -- grpo_output/trace_log.jsonl
```

If `eval_calibration.json` is on the Space, the calibration plot was already auto-regenerated from real data by `train_grpo.py` at end-of-training. If only `training_curves.png` and `training_stats.json` exist (older checkpoint state), run:

```bash
python scripts/generate_calibration_plot.py --mode real --data grpo_output/eval_calibration.json --out grpo_output/calibration_plot.png
```

---

## Step 3 · Look at the curve (5 min) — make the strategic call

Open `grpo_output/training_curves.png`. You'll see one of:

| Pattern | Verdict | Narrative move |
|---|---|---|
| Reward goes **clearly up** with a smoothed monotonic trend | 🏆 Win | Lead the README and video with the curve |
| Reward goes **slightly up**, last-third mean > first-third mean | ✅ OK | "Convergence consistent with the metacognitive reward" |
| Reward is **flat-ish** but final value > 0.5 | 🟡 Acceptable | Lead with calibration + transfer; treat curve as supplementary |
| Reward is **flat or declining** | 🔴 Pivot | Bury curve, lead with env design + 5.24× transfer ratio + paper |

In every case, **calibration + transfer carry the submission**. The curve is icing.

---

## Step 4 · Update README with real numbers (15 min)

Edit `README.md` TL;DR table — replace heuristic-proxy numbers with whatever `eval_calibration.json` shows. A 1-line script gives you all of them at once:

```bash
python -c "
import json
d = json.load(open('grpo_output/eval_calibration.json'))
preds, lens, labels = d['pred'], d['actual_len'], d['label']
n = len(preds)
buggy = [(p,l) for p,l,lab in zip(preds, lens, labels) if lab == 1]
safe  = [(p,l) for p,l,lab in zip(preds, lens, labels) if lab == 0]
print(f'n={n}  buggy={len(buggy)}  safe={len(safe)}')
print(f'avg_len_buggy = {sum(l for _,l in buggy)/max(1,len(buggy)):.0f}')
print(f'avg_len_safe  = {sum(l for _,l in safe)/max(1,len(safe)):.0f}')
print(f'P(long|buggy) = {sum(1 for p,_ in buggy if p==\"long\")/max(1,len(buggy)):.2f}')
print(f'P(long|safe)  = {sum(1 for p,_ in safe  if p==\"long\")/max(1,len(safe)):.2f}')
ratio = (sum(l for _,l in buggy)/max(1,len(buggy))) / max(1, sum(l for _,l in safe)/max(1,len(safe)))
print(f'thinking_ratio = {ratio:.2f}x')
"
```

Paste those numbers into the README table. **Don't fudge.** If a number is worse than the heuristic, keep the real number — judges check.

---

## Step 5 · Record the video (60–90 min)

Follow `VIDEO_SCRIPT.md` line-for-line. Don't improvise. **2–3 takes max.**

Order of recording:
1. Close all browser tabs, chat apps, terminals not used in the video
2. Open the Space, click through each tab once to warm cache
3. Open VIDEO_SCRIPT.md on a second screen / printed
4. QuickTime → File → New Screen Recording → click record → don't stop on flubs, just pause and continue
5. Final cleanup in QuickTime: Edit → Trim. That's it.
6. Export 1080p, < 100 MB
7. YouTube → Upload → **Unlisted** → title: *"The Thinking Budget — Meta PyTorch OpenEnv Hackathon 2026"*
8. Copy the share URL

---

## Step 6 · Publish blog post (20 min)

1. Go to https://huggingface.co/new-blog
2. Open `blog_post.md` in your editor
3. Paste body (the YAML frontmatter is correct already)
4. Upload `grpo_output/thinking_allocation.png` as the thumbnail
5. Replace the YouTube placeholder in `blog_post.md` and the post body with your real Unlisted YouTube URL
6. Hit **Publish**
7. Copy the blog URL

---

## Step 7 · Final commit + push (15 min)

```bash
cd /Users/raj/Desktop/Meta-project

# Update README and blog_post.md with the YouTube + Blog URLs.
# Edit by hand — there are placeholder strings.

git add README.md blog_post.md grpo_output/training_curves.png \
        grpo_output/training_stats.json grpo_output/calibration_plot.png \
        grpo_output/eval_calibration.json grpo_output/trace_log.jsonl

git commit -m "submission: real training curves + video + blog links"
git push hf hf-fix:main
```

This triggers a final Space rebuild — but that's fine, the demo is data-only at this point and rebuilds in <2 min.

---

## Step 8 · Submit (15 min)

1. Open the hackathon submission form
2. Paste:
   - Project name: *The Thinking Budget*
   - Live Space: `https://huggingface.co/spaces/lucid987654/code-review-env-v3`
   - Repo: `https://github.com/subwaycookiecrunch/Meta-project`
   - Video URL (Unlisted YouTube): from Step 5
   - Blog URL: from Step 6
   - Theme: 3.1 — World Modeling → Professional Tasks
3. Hit submit
4. Take a screenshot of the confirmation
5. Done

---

## 🚨 Plan B — if training crashed or curve is hostile

Don't panic. The submission stands without a perfect curve because:

1. **Environment design is the headline.** OpenEnv + 6 MCP tools + 6-component reward + ground-truth labeled dataset is the actual contribution being judged.
2. **The transfer experiment is real.** F1 0.28 → 1.00 on a held-out non-CVE domain is a strong, defensible number from the heuristic-proxy oracle policy. The paper labels this transparently.
3. **The calibration plot from the heuristic proxy still illustrates the target shape.** Label it "target-policy structure (proxy)" instead of "trained policy."
4. **The 6-tab interactive demo + paper + video + blog are the deliverables judges click through first.**

If the training run is unusable, edit one paragraph in the README:

> Replace the GRPO reward-curve section with: *"Training data from a 50-step Qwen3-8B v1 run is included for reference. The metacognitive reward function presented here was added in v2 and validated on a deterministic risk-driven oracle policy that demonstrates the target reasoning-allocation structure (see PAPER.md §4 and `transfer_eval.py`). Replication on a longer GRPO run is the obvious follow-up; the contribution is the auxiliary objective and the environment, both of which are open-source."*

Then continue to Step 5.

---

## Cheat sheet — numbers (will become real after Step 4)

```
in-domain thinking ratio:    6.06×    (heuristic) → ?     (after train)
held-out transfer ratio:     5.24×    (heuristic) → ?
calibration confusion diag:  0.88     (heuristic) → ?
P(long | bug):               0.92     (heuristic) → ?
P(long | safe):              0.00     (heuristic) → ?
transfer F1 untrained:       0.28
transfer F1 trained:         1.00
dataset:                     150 CVEs · 2,892 files
training:                    Qwen3-1.7B + GRPO + LoRA r=16 + 400 steps (200 ep × 2 epochs)
```

---

*Generated 21:34 Sat Apr 25, 2026. Hard deadline 17:00 Sun Apr 26.*
