import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code_review_env.models import CodeReviewAction
from code_review_env.server.environment import CodeReviewEnvironment, BUGGY_EPISODES


def run_episode(env, agent_fn, name, episode=None):
    if episode:
        orig = random.choice
        random.choice = lambda x: episode
        obs = env.reset()
        random.choice = orig
    else:
        obs = env.reset()

    total_r = 0.0
    steps = 0

    print(f"\n--- {name} ---")
    print(f"repo: {obs.repo_name} ({obs.cve_id}, cvss {obs.cvss_score})")
    print(f"files: {obs.total_files}, budget: {obs.review_budget}")

    while not obs.done:
        obs = env.step(agent_fn(obs))
        total_r += (obs.reward or 0.0)
        steps += 1

    print(obs.message)
    print(f"total: {total_r:.1f} ({steps} steps)")
    return obs


def random_agent(obs):
    return CodeReviewAction(decision=random.choice(["flag", "skip"]))

def threshold_agent(obs):
    s = obs.churn_score + obs.complexity_score + obs.todo_score
    if s > 50 or obs.todo_score > 20:
        return CodeReviewAction(decision="flag")
    return CodeReviewAction(decision="skip")

def conservative_agent(obs):
    w = obs.churn_score * 0.3 + obs.complexity_score * 0.4 + obs.todo_score * 0.2 + obs.recency_score * 0.1
    return CodeReviewAction(decision="flag" if w > 30 else "skip")

def aggressive_agent(obs):
    if obs.files_flagged < obs.review_budget:
        return CodeReviewAction(decision="flag")
    return CodeReviewAction(decision="skip")


if __name__ == "__main__":
    print("CodeReviewEnv demo")
    print("=" * 50)

    env = CodeReviewEnvironment()

    target = None
    if BUGGY_EPISODES:
        target = next((e for e in BUGGY_EPISODES if e["total_bugs"] >= 2), BUGGY_EPISODES[0])

    agents = [
        (random_agent, "random"),
        (threshold_agent, "threshold"),
        (conservative_agent, "conservative"),
        (aggressive_agent, "aggressive"),
    ]

    results = []
    for fn, name in agents:
        obs = run_episode(env, fn, name, episode=target)
        results.append((name, obs.f1_score, obs.precision, obs.recall, obs.true_positives, obs.false_positives, obs.false_negatives))

    print(f"\n{'='*60}")
    print(f"{'agent':<20} {'f1':>5} {'prec':>5} {'rec':>5} {'tp':>4} {'fp':>4} {'fn':>4}")
    print("-" * 60)
    for name, f1, p, r, tp, fp, fn in results:
        print(f"{name:<20} {f1:>5.2f} {p:>5.2f} {r:>5.2f} {tp:>4} {fp:>4} {fn:>4}")
