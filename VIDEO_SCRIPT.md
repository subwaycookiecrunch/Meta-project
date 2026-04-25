# 90-Second Video Recording Script

**Record this on Loom (free, no install) or QuickTime screen recording.**
**Upload to YouTube as unlisted. Add the URL to README.md.**

## Setup
- Open the HF Space in Chrome: https://huggingface.co/spaces/lucid987654/code-review-env-v3
- Make sure training is DONE (you should see training_curves.png)
- Full-screen Chrome, zoom to 100%

---

## Script (90 seconds total)

### [0:00-0:15] Hook — 15 seconds
> "LLMs waste compute thinking hard about easy problems. I trained a 1.7B model
> to predict how hard a file is BEFORE reasoning about it — and to allocate
> its thinking budget accordingly."

**Show:** The hero histogram tab (📊 The Thinking Budget)

### [0:15-0:35] The mechanism — 20 seconds
> "Before every think block, the model emits a budget prediction — short, medium, or long.
> The reward function scores three things: did the actual length match the prediction?
> Did the model predict 'long' on buggy files? And was every prediction followed by a real action?"

**Show:** Scroll through a trace in the 🧠 Try The Agent tab — point at the budget_prediction tags

### [0:35-0:55] Results — 20 seconds
> "After training: 6x thinking ratio between buggy and safe files.
> The same policy transfers to a completely different domain — non-security code review —
> with F1 going from 0.28 to 1.00."

**Show:** Click 📐 Calibration and Transfer tab, show the transfer plot

### [0:55-1:10] The defense — 15 seconds
> "I tried to break the reward with 5 different attack strategies.
> All of them score below the honest policy. The closest one still loses by 22 percent."

**Show:** Click 🛡 Red Team tab, show the attack comparison table

### [1:10-1:25] Training curve — 15 seconds
> "Here is the real training curve from a single A10G.
> Reward goes from near-zero to 0.4+ over 150 GRPO steps."

**Show:** Click 🏋️ Training Progress tab, show the curve

### [1:25-1:30] Close — 5 seconds
> "The code, paper, and red team writeup are all in the repo. Thanks."

---

## After recording
1. Upload to YouTube (unlisted)
2. Update README.md line with the video URL
3. Update JUDGES.md demo video row
4. git push
