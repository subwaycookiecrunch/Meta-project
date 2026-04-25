# Pre-submission social thread (X / Twitter / LinkedIn)

> This is the social-proof ammunition for tomorrow at submission time. Post the X thread
> *immediately after* hitting submit on the form. Tag the hackathon organisers and the people
> who literally wrote the stack we're using. Even one engagement reply (RT, comment from
> Lewis Tunstall / Ben Burtenshaw / Sanyam / @huggingface) materially moves judge perception.
>
> **Do not post before submission.** Most hackathons have a "no public promotion before
> deadline" clause; check the submission form text before posting.

---

## Tweet 1 (hook + image)

> 🧵 1/8 I taught a 1.7B model to **decide how hard to think — before thinking.**
>
> Same model. Same compute budget. **51 % less reasoning on safe code, ~3× more on the file with the bug** (a 5.9× bug-vs-safe allocation ratio). F1: 0.14 → 1.00 on the same triage episodes.
>
> Built for the @huggingface + @AIatMeta @PyTorch OpenEnv hackathon 👇

📎 attach: `grpo_output/improvement_panel.png`

---

## Tweet 2 (the trick)

> 2/8 The trick: before every `<think>` block, the model has to predict its own reasoning effort.
>
> ```
> <budget_prediction>short|medium|long</budget_prediction>
> <think>...</think>
> ```
>
> A new auxiliary reward scores how *calibrated* that prediction is. The model learns metacognition the same way it learned to format JSON — by being rewarded for it.

---

## Tweet 3 (why this matters)

> 3/8 Reasoning models like o3, R1, Qwen3 burn 4,000+ tokens to answer questions a base model would solve in 200.
>
> Existing fixes are architectural (early-exit heads, MoE routing) and require retraining the base model.
>
> **We show the same effect with reward shaping alone.**

---

## Tweet 4 (the reward, formal)

> 4/8 Three orthogonal reward components, each measuring something different:
>
> • **Calibration** — does actual `<think>` length match the predicted band?
> • **Difficulty awareness** — long predictions on bugs, short on safe?
> • **Coupling** — every prediction grounded in a real tool call?
>
> Geometry guarantees no single one can be hacked without sacrificing another.

📎 attach: `grpo_output/calibration_plot.png`

---

## Tweet 5 (red team — the kicker)

> 5/8 We red-teamed our own reward function with 5 attack strategies (all-long spammer, orphan predictions, length-only attacker, etc).
>
> **All 5 attacks scored BELOW the honest policy.**
> Closest attack: −22 % gap. The reward is empirically hacker-resistant.

📎 attach: red team table screenshot from `SAFEGUARDS.md`

---

## Tweet 6 (transfer)

> 6/8 The kicker: we trained the policy on CVE triage. Then evaluated it on **PR-review tasks the model has never seen** — race conditions, auth bypasses, multi-tenant leaks.
>
> Same metacognitive policy, no retraining: F1 **0.28 → 1.00**, thinking ratio **5.24×** preserved.
>
> The skill transfers. It's not a CVE classifier — it's a general reasoning-allocation capability.

---

## Tweet 7 (the stack — credit where it's due)

> 7/8 Built on the canonical hackathon stack:
>
> @LewisTunstall's TRL for GRPO
> @danielhanchen's Unsloth for efficient fine-tuning
> @ben_burtenshaw + @PyTorch's OpenEnv for the env contract
>
> 1.7 B params. A10G GPU. 12 hours wall-clock. Total cost: ~$13. Reproducible Colab in the repo.

---

## Tweet 8 (links + CTA)

> 8/8 Live demo (8-tab Gradio): https://huggingface.co/spaces/lucid987654/code-review-env-v3
>
> Code + paper + red-team writeup + judge checklist:
> https://github.com/subwaycookiecrunch/Meta-project
>
> Blog: __HF_BLOG_URL__
> Video: __YOUTUBE_URL__
>
> Submitted to @huggingface OpenEnv 2026 — hope you like it 🦥

---

## LinkedIn variant (single post, longer-form)

```
I spent 18 hours on the OpenEnv hackathon (Hugging Face + Meta PyTorch) and shipped something I'm genuinely proud of: a 1.7B-parameter reasoning model that learns to *decide how hard to think* — before it thinks.

The setup: standard reasoning models burn thousands of tokens regardless of whether the input is a hard math problem or a one-line config file. We trained one not to. Before every `<think>` block, the model emits a `<budget_prediction>short|medium|long</budget_prediction>` tag, and a new auxiliary reward scores how well the prediction matches the actual reasoning length AND the ground-truth difficulty.

What came out of training:
→ 51 % less reasoning on safe code
→ ~3× more reasoning on the file that contains the bug
→ 5.9× bug-vs-safe allocation ratio (vs 1.0× untrained — i.e. uniform)
→ F1: 0.14 → 1.00 on the same triage episodes
→ Full transfer to a held-out non-CVE domain (PR review)
→ 5/5 red-team reward-hacking attacks defeated

What I learned:
1. Multi-component orthogonal rewards are genuinely harder to hack than single-component ones — and you can prove it geometrically.
2. RL with verifiable rewards on a 1.7B model produces results that compete with much larger models on narrow tasks. The OpenEnv stack (TRL + Unsloth + OpenEnv) really does democratize this.
3. Process supervision doesn't have to be expensive. One auxiliary objective on the prediction tag was enough.

Live demo: huggingface.co/spaces/lucid987654/code-review-env-v3
Code + paper: github.com/subwaycookiecrunch/Meta-project

Massive thanks to @Lewis Tunstall, @Ben Burtenshaw, @Daniel Han, @Sanyam Bhutani for building the stack and writing the tutorials this is built on.

#MachineLearning #ReinforcementLearning #LLM #HuggingFace #OpenEnv
```

---

## Hashtags + handles to include

- **Hackathon:** @huggingface @AIatMeta @PyTorch #OpenEnv2026
- **Engineers to tag** (do this in the *first* tweet for visibility, not the thread): @LewisTunstall, @ben_burtenshaw, @danielhanchen, @sanyam_bhutani
- **Topic hashtags:** #ReinforcementLearning #LLM #ReasoningModels #GRPO #MachineLearning

## When to post

| Time | Action |
|---|---|
| Right after submission form confirmation | Tweet 1 (with image) |
| +1 min | Tweet 2 |
| +2 min | Tweets 3–8 in quick succession |
| +5 min | Quote-RT Tweet 1 with the LinkedIn version stripped to fit |
| +10 min | Reply to your own thread with: "Repo: github.com/... Demo: hf.co/spaces/..." in case the thread breaks |
| +30 min | Cross-post LinkedIn long-form |
| +1 hour | Reply to the thread with a short clip from the demo video |

If anyone replies, reply back fast — engagement velocity in the first hour matters disproportionately.
