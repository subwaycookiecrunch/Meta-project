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

# CodeReviewEnv

An RL environment for vulnerability triage, built on real CVE data from the NVD.

The idea: most RL envs are toy problems (gridworld, cartpole, etc). We wanted something closer to what devs actually deal with — triaging security patches across a codebase with limited time and attention.

The agent gets a stream of files from a real CVE patch and has to decide: **flag** this file for human review, or **skip** it. There's a fixed review budget so you can't just flag everything.

* **HF Space**: https://huggingface.co/spaces/lucid987654/code-review-env
* **GitHub**: https://github.com/subwaycookiecrunch/Meta-project

---

### Data

1715 files across 65 CVEs scraped from actual GitHub vulnerability patches. Each file has four features extracted from the commit history: churn, complexity, TODO count, and recency.

### Rewards

Asymmetric on purpose — missing a real bug is worse than wasting a review slot on a clean file.

| Outcome | Reward | Why |
|---------|--------|-----|
| True Positive | +1.0 | found a real bug |
| True Negative | +0.8 | correctly skipped clean file |
| False Positive | -0.4 | wasted budget on safe file |
| False Negative | -0.2 | missed a bug |
| Over-budget flag | -0.5 | budget is a hard limit |

### Tasks

Three difficulty levels based on repo size:

- **easy**: ≤15 files, generous budget
- **medium**: 16-29 files
- **hard**: 30+ files, tight budget — agent really has to pick its spots

Grading is F1 score (precision × recall), always in [0, 1].

---

### Observation fields

Each step gives you:
- `file_path`, `file_index`, `total_files`, `files_remaining`
- `churn_score`, `complexity_score`, `todo_score`, `recency_score`
- `review_budget`, `files_flagged`
- `difficulty_level`, `cve_id`, `repo_name`
- terminal: `precision`, `recall`, `f1_score`, `true_positives`, `false_positives`, etc.

Action is just `{"decision": "flag"}` or `{"decision": "skip"}`.

---

## Running it

**Install:**
```bash
pip install openenv-core openai
```

**Docker:**
```bash
docker build -t codereviewenv .
docker run -p 7860:7860 codereviewenv
```

**Inference:**
```bash
export HF_TOKEN="your_token"
python inference.py
```

---

## Agents

### LLM baseline (`inference.py`)

Sends the file stats to Qwen2.5-Coder-32B via the HF inference API and asks it to flag or skip. Runs all three difficulty levels.

Rough zero-shot numbers:

| Difficulty | F1 | Precision | Recall |
|---|---|---|---|
| Easy | ~0.15 | ~0.12 | ~0.25 |
| Medium | ~0.10 | ~0.08 | ~0.18 |
| Hard | ~0.08 | ~0.06 | ~0.15 |

Lots of room to improve — the LLM has no training signal, it's just guessing from feature names.

```bash
export HF_TOKEN="your_token"
python inference.py
```

### PyTorch agent (`train_pytorch_agent.py`)

REINFORCE with a 3-layer MLP. Takes the 6 observation features as input, outputs flag/skip probabilities. Trains directly against the env reward signal.

```bash
pip install torch
python train_pytorch_agent.py
```

## File layout

- `Dockerfile` + `openenv.yaml` — deployment config
- `inference.py` — LLM baseline (hackathon submission script)
- `train_pytorch_agent.py` — pytorch RL agent
- `models.py` — pydantic action/observation/state types
- `server/environment.py` — core env logic + reward math
- `data/` — the CVE dataset

MIT License
