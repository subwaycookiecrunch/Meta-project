---
title: CodeReviewEnv
emoji: 🛡️
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
---

# CodeReviewEnv

RL environment for training agents to triage code files for vulnerabilities. Built on top of OpenEnv (Meta/PyTorch).

Uses real labeled data from NVD - 1715 file samples across 65 CVEs. The agent reviews files from a repo and has to figure out which ones are actually buggy based on features like churn, complexity, TODO count, and recency.

## How it works

Each episode loads files from a real CVE-affected codebase. The agent goes through files one by one and decides `flag` or `skip`. At the end it gets scored on precision/recall/f1.

Rewards are intentionally asymmetric - missing a real bug hurts way more than flagging a false positive, because thats how it works in practice.

| outcome | reward |
|---------|--------|
| correctly flag a bug | 1.0 |
| correctly skip safe file | 0.8 |
| flag a safe file | 0.4 |
| miss a real bug | 0.0 |

The agent also has a limited flag budget per episode so it can't just flag everything.

## Setup

```bash
pip install openenv-core openai
```

To run the server locally:
```bash
docker build -t codereviewenv .
docker run -p 8000:8000 codereviewenv
```

## Client usage

To run the baseline evaluation script using an LLM:
```bash
export HF_TOKEN="your_token"
python inference.py
```

### PyTorch Reinforcement Learning Agent
Since LLMs often struggle natively with our asymmetric reward mechanism (scoring 0.00 F1 on zero-shot baselines), we've implemented a native **PyTorch Deep Reinforcement Learning Agent** using Policy Gradients (REINFORCE). 

The agent learns to allocate its flag budget optimally across the codebase by backpropagating on the OpenEnv step rewards:
```bash
python train_pytorch_agent.py
```

## Observation fields

- `file_path` - current file being reviewed
- `churn_score` / `complexity_score` / `todo_score` / `recency_score` - features (0-100)
- `review_budget` - how many flags left
- `files_remaining` - files left in episode
- `cve_id` / `cvss_score` - which vulnerability this episode is about

## Data

The training data comes from DeathClock's CVE pipeline:
1. Pull CVEs from NVD API with GitHub refs
2. Fetch file metadata and commit history from those repos
3. Compute feature scores for each file
4. Label files based on which ones were actually patched in the CVE fix

65 CVEs, 1715 files total. 12 confirmed buggy files across 7 episodes - rest are negative examples that the agent needs to learn to skip.

## Files

```
code_review_env/
├── Dockerfile         - project root dockerfile (for openenv)
├── inference.py       - generic llm evaluator script
├── train_pytorch_agent.py - native PyTorch DRL training loop
├── models.py          - action/observation/state types
├── client.py          - websocket client
├── demo.py            - runs baseline agents locally without docker
├── data/              - cve training data (json)
└── server/
    ├── environment.py - core env logic
    └── app.py         - fastapi server
```

## License

MIT
