---
title: CodeReviewEnv
emoji: 🛡️
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
---

# CodeReviewEnv: Triage CVEs Like a Pro

*Built for the Meta/PyTorch OpenEnv Hackathon*

Most RL environments are toy setups like GridWorld. We wanted to tackle a real problem: **Vulnerability Triage**. 

We built `CodeReviewEnv` using real-world data from the National Vulnerability Database (NVD). The agent connects to a repository, scans the files, and uses heuristics (like codebase churn, complexity, and recency) to decide whether to `flag` a file for manual security review or `skip` it.

### The Problem
We have 1715 files across 65 CVEs pulled directly from actual GitHub patches. 
We hit the agent with **Asymmetric Rewards**. In the real world, missing a critical bug (False Negative) is infinitely worse than accidentally flagging a safe file (False Positive). 

Our reward table forces the agent to balance its paranoia:
* Found a bug: +1.0
* Correctly skipped a safe file: +0.8
* Flagged a safe file: +0.4
* Missed a real bug: 0.0

Oh, and there's a strict **Review Budget**. You can't just flag everything, or you run out of budget. 

## Setup & Running

**1. Install deps:**
```bash
pip install openenv-core openai torch
```

**2. Spin up the FastAPI Server:**
```bash
docker build -t codereviewenv .
docker run -p 8000:8000 codereviewenv
```
*(If you are viewing this on Hugging Face Spaces, the server is automatically running securely!)*

## The Agents (We built two!)

### 1. The Zero-Shot LLM Baseline (`inference.py`)
This is the standard OpenEnv submission script required by the Hackathon. We wrote a wrapper that passes the environment state into any OpenAI-compatible LLM to see if a huge model can reason through the file stats to allocate its budget.
```bash
export HF_TOKEN="your_huggingface_write_key"
python inference.py
```

### 2. The Native PyTorch Agent (`train_pytorch_agent.py`)
**Flex Warning.** LLMs are cool, but they completely suck at understanding strict mathematical bounds (like rationing a flag budget over exactly 65 files with precise asymmetric scoring). So, we went bare-metal.

We built a custom Deep Reinforcement Learning Agent using native PyTorch Policy Gradients (REINFORCE) to interface perfectly with the OpenEnv API. It iteratively converges to find the perfect risk/reward strategy.
```bash
python train_pytorch_agent.py
```

## Hackathon Repo Tour
* Dockerfile & openenv.yaml: The OpenEnv backend deployment wrappers
* inference.py: The mandatory LLM endpoint validation script
* train_pytorch_agent.py: Our custom PyTorch REINFORCE brain
* /server/environment.py: Where the magic reward mathematics happen
* /data/: The actual scraped CVE GitHub dataset 

MIT License. Thanks for checking it out!
