from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from .models import CodeReviewAction, CodeReviewObservation, CodeReviewState


class CodeReviewEnv(EnvClient[CodeReviewAction, CodeReviewObservation, CodeReviewState]):

    def _step_payload(self, action: CodeReviewAction) -> dict:
        return {"decision": action.decision}

    def _parse_result(self, payload: dict) -> StepResult:
        obs = payload.get("observation", {})
        done = payload.get("done", False)
        reward = payload.get("reward")

        return StepResult(
            observation=CodeReviewObservation(
                done=done, reward=reward,
                file_path=obs.get("file_path", ""),
                file_index=obs.get("file_index", 0),
                total_files=obs.get("total_files", 0),
                churn_score=obs.get("churn_score", 0.0),
                complexity_score=obs.get("complexity_score", 0.0),
                todo_score=obs.get("todo_score", 0.0),
                recency_score=obs.get("recency_score", 0.0),
                cve_id=obs.get("cve_id", ""),
                cvss_score=obs.get("cvss_score", 0.0),
                repo_name=obs.get("repo_name", ""),
                files_remaining=obs.get("files_remaining", 0),
                files_flagged=obs.get("files_flagged", 0),
                review_budget=obs.get("review_budget", 0),
                message=obs.get("message", ""),
                true_positives=obs.get("true_positives", 0),
                false_positives=obs.get("false_positives", 0),
                false_negatives=obs.get("false_negatives", 0),
                true_negatives=obs.get("true_negatives", 0),
                precision=obs.get("precision", 0.0),
                recall=obs.get("recall", 0.0),
                f1_score=obs.get("f1_score", 0.0),
            ),
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: dict) -> CodeReviewState:
        return CodeReviewState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            cve_id=payload.get("cve_id", ""),
            repo_name=payload.get("repo_name", ""),
            total_files=payload.get("total_files", 0),
            total_bugs=payload.get("total_bugs", 0),
            current_file_index=payload.get("current_file_index", 0),
            files_flagged=payload.get("files_flagged", 0),
            correct_flags=payload.get("correct_flags", 0),
            review_budget=payload.get("review_budget", 0),
        )
