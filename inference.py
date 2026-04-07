import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from code_review_env import CodeReviewEnv, CodeReviewAction

_raw_api_base = os.environ.get("API_BASE_URL")
if _raw_api_base and "8000" in _raw_api_base:
    API_BASE_URL = "https://router.huggingface.co/v1"
else:
    API_BASE_URL = _raw_api_base or "https://router.huggingface.co/v1"

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-Coder-32B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN", os.environ.get("API_KEY"))
LOCAL_IMAGE_NAME = os.environ.get("LOCAL_IMAGE_NAME")

TASK_NAME = "code_review"
BENCHMARK = "code_review_env"

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def main():
    if not HF_TOKEN:
        print("Please set the HF_TOKEN environment variable.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    
    env_url = os.environ.get("ENV_SERVER_URL", "http://127.0.0.1:8000")
    
    rewards: List[float] = []
    steps_taken = 0
    success = False
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        with CodeReviewEnv(base_url=env_url).sync() as env:
            step_result = env.reset()
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
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=10,
                        temperature=0.1
                    )
                    ans = res.choices[0].message.content.strip().lower()
                    decision = "flag" if "flag" in ans else "skip"
                except Exception as e:
                    error_msg = str(e).replace('\n', ' ')

                action = CodeReviewAction(decision=decision)
                step_result = env.step(action)
                obs = step_result.observation
                
                reward = step_result.reward or 0.0
                rewards.append(reward)
                done = step_result.done
                
                log_step(step=steps_taken, action=decision, reward=reward, done=done, error=error_msg)
            
            score = obs.f1_score if hasattr(obs, 'f1_score') else 0.0
            
            score = min(max(score, 0.0), 1.0)
            
            success = score > 0.0

    except Exception as e:
        error_msg = str(e).replace('\n', ' ')
        print(f"[DEBUG] Execution error: {error_msg}", file=sys.stderr)
        score = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    main()
