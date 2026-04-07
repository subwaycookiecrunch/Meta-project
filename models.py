from openenv.core.env_server import Action, Observation, State


class CodeReviewAction(Action):
    decision: str


class CodeReviewObservation(Observation):
    file_path: str = ""
    file_index: int = 0
    total_files: int = 0
    difficulty_level: str = "medium"
    churn_score: float = 0.0
    complexity_score: float = 0.0
    todo_score: float = 0.0
    recency_score: float = 0.0
    cve_id: str = ""
    cvss_score: float = 0.0
    repo_name: str = ""
    files_remaining: int = 0
    files_flagged: int = 0
    review_budget: int = 0
    message: str = ""
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0


class CodeReviewState(State):
    cve_id: str = ""
    repo_name: str = ""
    total_files: int = 0
    total_bugs: int = 0
    current_file_index: int = 0
    difficulty_level: str = "medium"
    files_flagged: int = 0
    correct_flags: int = 0
    review_budget: int = 0
