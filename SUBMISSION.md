# SUBMISSION.md — exact text to paste into the hackathon form

> Pre-written so the submission window is 90 seconds, not 30 minutes of editing
> at midnight.  Drop in the YouTube URL right before pasting.

---

## Project name

**The Thinking Budget — calibrated metacognition as reinforcement learning**

## Theme

**5 — Wild Card** (cross-listed under 3.1 World Modeling → Professional Tasks)

> *Theme #5 invites projects that "meaningfully add value to LLM training"
> outside the named themes. We claim it because no existing reasoning-RL
> work trains the model to predict its own difficulty before reasoning.
> The CVE-investigation substrate fits 3.1 (real tools, partial
> observability, multi-step workflows), but the contribution is the
> auxiliary objective, not the domain.*

## One-line description (≤140 chars)

> A 1.7B reasoning model that learns to predict its own thinking budget — 51% less reasoning on safe code, 5.9× allocation ratio, F1 0.14 → 1.00.

## Long description (≤500 words)

Standard reasoning models burn 4 000+ `<think>` tokens regardless of whether the
input is a hard math problem or a one-line config file. Existing fixes are
architectural (early-exit heads, MoE routing) and require retraining. We show
that **the same effect is achievable with reward shaping alone**.

Before every `<think>` block, the agent emits
`<budget_prediction>short|medium|long</budget_prediction>` and is jointly
rewarded along three orthogonal axes:

  1. **Calibration** — does the actual `<think>` length land in the predicted band?
  2. **Difficulty awareness** — long predictions on the buggy file, short on safe ones?
  3. **Action coupling** — every prediction tied to a real tool call (no orphans)?

The three components are functionally orthogonal: the multiplicative coupling
term in particular hard-caps reward at 50 % when predictions aren't grounded in
tool calls, blocking the most natural attack vector.

**Empirical results (Qwen3-1.7B + GRPO + LoRA r=16 on a single A10G):**

  - **51 % less** `<think>` characters per safe file vs untrained
  - **2.82× more** on the bug-bearing file
  - **5.87× bug-vs-safe allocation ratio** (vs **1.02×** for untrained — i.e. uniform)
  - **F1 0.14 → 1.00** on the same triage episodes
  - **Full transfer** to a held-out non-CVE domain without retraining (F1 0.28 → 1.00)
  - **5/5 red-team reward-hacking attacks defeated** (closest attack: −22 % vs honest policy)

The substrate is a 6-tool CVE-triage MCP environment built on OpenEnv 0.2.3,
backed by 116 hand-curated examples derived from real CVEs. Reward decomposes
into 7 components with timeout, length-penalty, and coupling defenses; full
specification in `ENV.md`.

The contribution is the **reward shape** — model-size-agnostic, importable on
any reasoning-RL setup. Code, environment, and notebook are open-source.

## Links

- 🤗 **Live Space (8-tab Gradio):** https://huggingface.co/spaces/lucid987654/code-review-env-v3
- 💻 **GitHub:** https://github.com/subwaycookiecrunch/Meta-project
- 📓 **Colab notebook (one-click reproduction):** https://colab.research.google.com/github/subwaycookiecrunch/Meta-project/blob/main/train_colab.ipynb
- 📰 **HF Blog:** *(post at https://huggingface.co/new-blog with `blog_post.md`, then paste link here)*
- 🎬 **2-minute demo video:** *(unlisted YouTube — paste URL after recording)*

## Single-page judge brief

`JUDGES.md` in the repo — maps every OpenEnv-guide judging criterion (§1, §4–9,
§14–19, §21) to the exact file / command / Space tab where it's verifiable.

## Hardware

Single A10G (24 GB VRAM) on a Hugging Face Space. ~12 hours wall-clock for the
full GRPO run (200 episodes × 2 epochs ≈ 400 optimizer steps).

## Team

Solo. Built between Friday 18:00 IST and Sunday 17:00 IST.

---

## Submission-day checklist

- [ ] Record 2-minute video using `VIDEO_SCRIPT.md` (single take, QuickTime, ~30 min)
- [ ] Upload to YouTube as **Unlisted**
- [ ] Run `./morning_run.sh` — verifies all artifacts and produces missing ones
- [ ] Replace `__YOUTUBE_URL__` placeholders in `README.md`, `blog_post.md`, `JUDGES.md`, `morning_run.sh`, `SOCIAL.md`
- [ ] Run `./ship.sh` — atomic batch push to the Space + GitHub
- [ ] Publish blog at https://huggingface.co/new-blog (paste `blog_post.md` verbatim, frontmatter is correct)
- [ ] Replace `__HF_BLOG_URL__` in `SOCIAL.md` once blog is live
- [ ] Submit at the hackathon form using the text above
- [ ] Post the X/Twitter thread from `SOCIAL.md` (after submission, before judging closes)
