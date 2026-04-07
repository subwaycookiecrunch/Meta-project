import json
import random
import uuid
import os
from collections import defaultdict
from openenv.core.env_server import Environment
from code_review_env.models import CodeReviewAction, CodeReviewObservation, CodeReviewState


DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "src", "data", "cveTrainingData.json"
)
FALLBACK_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data", "cve_training_data.json"
)


def _load_episodes():
    path = DATA_PATH if os.path.exists(DATA_PATH) else FALLBACK_DATA_PATH
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    groups = defaultdict(list)
    for s in raw:
        groups[(s["cveId"], s["repo"])].append(s)

    eps = []
    for (cve_id, repo), files in groups.items():
        buggy = [f for f in files if f["label"] == 1]
        if not buggy and len(files) > 30:
            files = random.sample(files, 30)
        eps.append({
            "cve_id": cve_id,
            "cvss": files[0].get("cvss", 0.0),
            "repo": repo,
            "files": files,
            "total_bugs": len(buggy),
        })
    return eps


EPISODES = _load_episodes()
BUGGY_EPISODES = [e for e in EPISODES if e["total_bugs"] > 0]
CLEAN_EPISODES = [e for e in EPISODES if e["total_bugs"] == 0]

try:
    n_bugs = sum(e["total_bugs"] for e in EPISODES)
    print(f"CodeReviewEnv: {len(EPISODES)} episodes, {len(BUGGY_EPISODES)} with bugs ({n_bugs} buggy files)")
except UnicodeEncodeError:
    print(f"CodeReviewEnv: {len(EPISODES)} episodes loaded")


class CodeReviewEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    TP_REWARD = 1.0
    FP_PENALTY = 0.4
    TN_REWARD = 0.8
    FN_PENALTY = 0.0

    def __init__(self):
        self._state = CodeReviewState()
        self._files = []
        self._idx = 0
        self._flagged = set()
        self._bugs = set()
        self._cum_reward = 0.0
        self._budget = 0

    def reset(self, seed=None, episode_id=None, **kwargs) -> CodeReviewObservation:
        if seed is not None:
            random.seed(seed)

        if BUGGY_EPISODES and random.random() < 0.7:
            ep = random.choice(BUGGY_EPISODES)
        else:
            ep = random.choice(EPISODES)

        self._files = list(ep["files"])
        random.shuffle(self._files)
        self._idx = 0
        self._flagged = set()
        self._bugs = {f["file"] for f in self._files if f["label"] == 1}
        self._cum_reward = 0.0
        self._budget = min(len(self._files), max(ep["total_bugs"] * 2 + 3, 5))

        self._state = CodeReviewState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0, cve_id=ep["cve_id"], repo_name=ep["repo"],
            total_files=len(self._files), total_bugs=ep["total_bugs"],
            current_file_index=0, files_flagged=0, correct_flags=0,
            review_budget=self._budget,
        )

        f = self._files[0]
        feat = f.get("features", [0, 0, 0, 0])
        return CodeReviewObservation(
            done=False, reward=None,
            file_path=f["file"], file_index=0, total_files=len(self._files),
            churn_score=float(feat[0]), complexity_score=float(feat[1]),
            todo_score=float(feat[2]), recency_score=float(feat[3]),
            cve_id=ep["cve_id"], cvss_score=float(ep["cvss"]), repo_name=ep["repo"],
            files_remaining=len(self._files) - 1, files_flagged=0,
            review_budget=self._budget,
            message=f"reviewing {ep['repo']} ({ep['cve_id']}) - {len(self._files)} files, budget {self._budget}",
        )

    def step(self, action: CodeReviewAction, timeout_s=None, **kwargs) -> CodeReviewObservation:
        decision = action.decision.lower().strip()
        if decision not in ("flag", "skip"):
            return CodeReviewObservation(
                done=False, reward=0.0,
                message=f"bad action: {decision}",
                file_path=self._files[self._idx]["file"],
                file_index=self._idx, total_files=len(self._files),
                files_remaining=len(self._files) - self._idx - 1,
                files_flagged=len(self._flagged), review_budget=self._budget,
            )

        self._state.step_count += 1
        cur = self._files[self._idx]
        is_bug = cur["file"] in self._bugs
        r = 0.0
        msg = ""

        if decision == "flag":
            if len(self._flagged) >= self._budget:
                r = -0.5
                msg = f"over budget, cant flag {cur['file']}"
            else:
                self._flagged.add(cur["file"])
                self._state.files_flagged += 1
                if is_bug:
                    r = self.TP_REWARD
                    self._state.correct_flags += 1
                    msg = f"hit - {cur['file']} is vulnerable"
                else:
                    r = self.FP_PENALTY
                    msg = f"miss - {cur['file']} was safe"
        else:
            if is_bug:
                r = self.FN_PENALTY
                msg = f"MISSED {cur['file']}"
            else:
                r = self.TN_REWARD
                msg = f"ok, skipped {cur['file']}"

        self._cum_reward += r
        self._idx += 1
        self._state.current_file_index = self._idx
        done = self._idx >= len(self._files)

        tp = len(self._flagged & self._bugs)
        fp = len(self._flagged - self._bugs)
        fn = len(self._bugs - self._flagged)
        tn = len(self._files) - tp - fp - fn

        if done:
            if self._bugs and self._bugs.issubset(self._flagged):
                msg += " +++ all bugs found"

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            msg += (f"\n\ndone ({self._state.cve_id}): "
                    f"p={prec:.2f} r={rec:.2f} f1={f1:.2f} "
                    f"tp={tp} fp={fp} fn={fn} tn={tn} "
                    f"reward={self._cum_reward:.1f}")

            return CodeReviewObservation(
                done=True, reward=r, file_path="",
                file_index=self._idx, total_files=len(self._files),
                files_remaining=0, files_flagged=len(self._flagged),
                review_budget=self._budget,
                cve_id=self._state.cve_id, cvss_score=0.0,
                repo_name=self._state.repo_name, message=msg,
                true_positives=tp, false_positives=fp,
                false_negatives=fn, true_negatives=tn,
                precision=prec, recall=rec, f1_score=f1,
            )

        nxt = self._files[self._idx]
        feat = nxt.get("features", [0, 0, 0, 0])
        return CodeReviewObservation(
            done=False, reward=r,
            file_path=nxt["file"], file_index=self._idx,
            total_files=len(self._files),
            churn_score=float(feat[0]), complexity_score=float(feat[1]),
            todo_score=float(feat[2]), recency_score=float(feat[3]),
            cve_id=self._state.cve_id, cvss_score=0.0,
            repo_name=self._state.repo_name,
            files_remaining=len(self._files) - self._idx - 1,
            files_flagged=len(self._flagged),
            review_budget=self._budget, message=msg,
        )

    @property
    def state(self) -> CodeReviewState:
        return self._state
