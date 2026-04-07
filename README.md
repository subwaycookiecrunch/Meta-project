---
title: CodeReviewEnv
emoji: 🛡️
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
tags:
  - openenv
---

# CodeReviewEnv: Triage CVEs Like a Pro

*Built for the Meta/PyTorch OpenEnv Hackathon*

Hey! 👋 Welcome to **CodeReviewEnv**. 

Most RL environments are toy setups like GridWorld or simple mazes. We wanted to tackle a real problem that we actually face as developers: **Vulnerability Triage**. 

We built `CodeReviewEnv` using real-world data scraped directly from patches in the National Vulnerability Database (NVD). The agent connects to a repository, scans the files, and uses heuristics (like codebase churn, complexity, and recency) to decide whether to `flag` a file for manual security review or `skip` it and move on.

* 🚀 **Hugging Face Space (Live Environment)**: [https://huggingface.co/spaces/lucid987654/code-review-env](https://huggingface.co/spaces/lucid987654/code-review-env)
* 📁 **GitHub Repository**: [https://github.com/subwaycookiecrunch/Meta-project](https://github.com/subwaycookiecrunch/Meta-project)

---

### The Problem

We have 1715 files across 65 authentic CVEs pulled directly from actual GitHub vulnerabilities. 
We hit the agent with **Asymmetric Rewards**. In the real world, missing a critical bug (False Negative) is infinitely worse than accidentally flagging a safe file for review (False Positive). 

Our reward table forces the agent to balance its paranoia:

| Outcome | Reward | Rationale |
|---------|--------|-----------|
| True Positive (found a real bug) | **+1.0** | Highest reward — catching vulnerabilities is the goal |
| True Negative (correctly skipped safe file) | **+0.8** | Good judgment that saves review budget |
| False Positive (flagged a safe file) | **-0.4** | Wastes review budget, penalized |
| False Negative (missed a real bug) | **-0.2** | Worst failure — penalty provides learning signal |
| Over-budget flag attempt | **-0.5** | Hard constraint — budget is non-negotiable |

Oh, and there's a strict **Review Budget**. You can't just flag everything, or you run out of budget and get heavily penalized!

---

### 🧩 Environment Specifications

#### Action Space
The action space is a straightforward, string-based categorical action:
* `decision`: `"flag"` or `"skip"`

#### Observation Space
At each step, the environment provides a rich state vector. The key metrics include:
* `file_path` & tracking metrics (`file_index`, `files_remaining`, `total_files`)
* **Difficulty & Context**: `difficulty_level`, `cve_id`, `repo_name`
* **Static Analysis Features**: 
  * `churn_score` — lines changed in the file (higher = more volatile)
  * `complexity_score` — cyclomatic complexity proxy (higher = harder to review)
  * `todo_score` — count of TODOs/FIXMEs (higher = more tech debt)
  * `recency_score` — how recently the file was modified (higher = more recent)
* **Limits**: `review_budget` and `files_flagged`
* **Terminal Metrics**: `precision`, `recall`, `f1_score`, `true_positives`, `false_positives`, `false_negatives`, `true_negatives`

#### 🎯 The Three Tasks (Difficulty Tiers)
We've partitioned the environment into three distinct difficulty tasks, scaling gracefully by the size of the repository logic the agent needs to parse over its fixed review budget:
1. **Easy (`difficulty="easy"`)**: Small repositories and pull requests (≤ 15 files). The budget is relatively forgiving.
2. **Medium (`difficulty="medium"`)**: Average-sized PRs (16-29 files) requiring more scrutiny.
3. **Hard (`difficulty="hard"`)**: Large-scale patches (30+ files). The agent is strapped for budget and must be extremely selective about utilizing its flags.

Each task includes a **programmatic grader** that returns a score between 0.0 and 1.0 (the F1-score), combining precision and recall into a single metric. The grader is deterministic and reproducible.

---

## Setup & Running

**1. Install deps:**
```bash
pip install openenv-core openai
```

**2. Spin up the FastAPI Server via Docker:**
```bash
docker build -t codereviewenv .
docker run -p 7860:7860 codereviewenv
```
*(If you are viewing this on Hugging Face Spaces, the server is automatically running!)*

**3. Run the inference script:**
```bash
export HF_TOKEN="your_huggingface_token"
python inference.py
```

---

## The Agents (We built two!)

### 1. The Zero-Shot LLM Baseline (`inference.py`)
This is the standard OpenEnv submission script required by the Hackathon. We wrote a wrapper that passes the environment state into an OpenAI-compatible LLM to see if a huge model can reason through the file stats to allocate its budget. It evaluates the environment sequentially over the **Easy**, **Medium**, and **Hard** tasks.

**Baseline Scores:**

| Difficulty | Model | F1-Score | Precision | Recall |
|-----------|-------|----------|-----------|--------|
| Easy | Qwen2.5-Coder-32B | ~0.15 | ~0.12 | ~0.25 |
| Medium | Qwen2.5-Coder-32B | ~0.10 | ~0.08 | ~0.18 |
| Hard | Qwen2.5-Coder-32B | ~0.08 | ~0.06 | ~0.15 |

These are zero-shot scores — the LLM has no training on this specific task, demonstrating there is real room for improvement via RL training.

```bash
export HF_TOKEN="your_huggingface_token"
python inference.py
```

### 2. The Native PyTorch Agent (`train_pytorch_agent.py`)
**Flex Warning.** LLMs are cool, but they struggle to implicitly understand strict mathematical bounds (like rationing a flag budget perfectly over exactly 29 files with precise asymmetric scoring). So, we went bare-metal.

We built a custom Deep Reinforcement Learning Agent using native PyTorch Policy Gradients (REINFORCE) to interface perfectly with the OpenEnv API. It iteratively converges to find the perfect risk/reward strategy.
```bash
pip install torch
python train_pytorch_agent.py
```

## Hackathon Repo Tour
* `Dockerfile` & `openenv.yaml`: The OpenEnv backend deployment wrappers
* `inference.py`: The mandatory LLM endpoint validation script
* `train_pytorch_agent.py`: Our custom PyTorch REINFORCE brain
* `/models.py`: Pydantic Models for Actions & Observations
* `/server/environment.py`: Where the magic reward mathematics happen
* `/data/`: The actual scraped CVE GitHub dataset 

*MIT License. Thanks for checking it out!*
