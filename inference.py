import os
import sys
from typing import List, Optional

from openai import OpenAI
from code_review_env import CodeReviewEnv, CodeReviewAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

TASK_NAME = "code_review"
BENCHMARK = "code_review_env"


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


def parse_decision(text: str) -> str:
    text = (text or "").strip().lower()
    return "flag" if text == "flag" else "skip"


def main():
    env_url = os.getenv("ENV_SERVER_URL", "http://127.0.0.1:8000")
    task_difficulty = "easy"

    rewards: List[float] = []
    steps_taken = 0
    success = False

    log_start(task=f"{TASK_NAME}_{task_difficulty}", env=BENCHMARK, model=MODEL_NAME)

    try:
        with CodeReviewEnv(base_url=env_url).sync() as env:
            step_result = env.reset(difficulty=task_difficulty)
            obs = step_result.observation

            while not step_result.done:
                steps_taken += 1
                budget_left = obs.review_budget - obs.files_flagged
                prompt = (
                    f"You are a code review assistant. Triaging file: {obs.file_path}\n"
                    f"Metrics -> churn: {obs.churn_score}, complexity: {obs.complexity_score}, "
                    f"todos: {obs.todo_score}, recency: {obs.recency_score}.\n"
                    f"Flag budget remaining: {budget_left}.\n"
                    f"Should we 'flag' or 'skip' this file? Answer exactly with 'flag' or 'skip'."
                )

                decision = "skip"
                error_msg = None

                try:
                    res = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": "You are a helpful code review assistant."},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=10,
                        temperature=0.1,
                    )
                    content = ""
                    if res.choices and res.choices[0].message and res.choices[0].message.content:
                        content = res.choices[0].message.content
                    decision = parse_decision(content)
                except Exception as e:
                    error_msg = str(e).replace("\n", " ")
                    print(error_msg, file=sys.stderr, flush=True)

                action = CodeReviewAction(decision=decision)
                step_result = env.step(action)
                obs = step_result.observation

                reward = step_result.reward or 0.0
                rewards.append(reward)

                log_step(
                    step=steps_taken,
                    action=decision,
                    reward=reward,
                    done=step_result.done,
                    error=error_msg,
                )

            success = bool(getattr(obs, "f1_score", 0.0) > 0.0)

    except Exception as e:
        print(str(e).replace("\n", " "), file=sys.stderr, flush=True)
        success = False

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)


if __name__ == "__main__":
    main()
