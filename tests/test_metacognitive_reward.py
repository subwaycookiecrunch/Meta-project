"""
tests/test_metacognitive_reward.py
===================================
Property-style tests for the metacognitive reward function.

These tests are the contract that the `<budget_prediction>` / `<think>` /
`<tool_call>` shape must satisfy to push GRPO into the target allocation
policy.  They double as the "honest policy" reference used by the red
team simulator (`scripts/red_team.py`): every adversarial completion
must score *strictly below* the honest policy on at least one
component.

Run from the repo root:

    pytest tests/ -q

Or directly:

    python -m unittest tests.test_metacognitive_reward
"""
from __future__ import annotations

import os
import sys
import unittest

# Allow tests to import from the repo root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from metacognitive_reward import (
    BAND_RANGES,
    METACOG_SYSTEM_PROMPT_ADDENDUM,
    compute_metacognitive_reward,
)


def _block(pred: str, think: str, tool: str, file_path: str) -> str:
    return (
        f"<budget_prediction>{pred}</budget_prediction>\n"
        f"<think>\n{think}\n</think>\n"
        f'<tool_call>{{"name": "{tool}", "arguments": {{"file_path": "{file_path}", '
        f'"reasoning": "x"}}}}</tool_call>\n'
    )


# ── Honest reference policies ──────────────────────────────────────────────
HONEST_LONG_BUG = _block(
    "long",
    "Looking at copy_from_user without a length check on the user-supplied size "
    "parameter. This is a textbook integer overflow into heap allocation. "
    "Function do_ioctl_handler at line 412 takes a size_t from userspace and "
    "feeds it into kmalloc without bounds checking; the CVE description "
    "matches exactly.",
    "flag_vulnerable",
    "drivers/foo.c",
)

HONEST_SHORT_SAFE = _block(
    "short",
    "Header file. No logic.",
    "skip_file",
    "include/types.h",
)


class CalibrationContract(unittest.TestCase):
    """The reward function must reflect the basic shape contract."""

    def test_honest_long_on_bug_scores_high(self):
        text = HONEST_LONG_BUG
        r = compute_metacognitive_reward(text, bug_files={"drivers/foo.c"})
        self.assertEqual(r.n_predictions, 1)
        self.assertAlmostEqual(r.calibration, 1.0, places=3)
        self.assertAlmostEqual(r.difficulty_awareness, 1.0, places=3)
        self.assertAlmostEqual(r.coupling, 1.0, places=3)
        self.assertGreater(r.raw_score, 0.95)

    def test_honest_short_on_safe_scores_high(self):
        # Note: difficulty_awareness only fires when bug_files is non-empty
        # (so the function knows "this safe file is intentionally safe in
        # an episode that has bugs elsewhere"; otherwise the label is
        # `None` and difficulty defaults to neutral 0.5).
        text = HONEST_SHORT_SAFE
        r = compute_metacognitive_reward(
            text, bug_files={"drivers/foo.c"}  # bugs exist elsewhere in episode
        )
        self.assertAlmostEqual(r.calibration, 1.0, places=3)
        self.assertAlmostEqual(r.difficulty_awareness, 1.0, places=3)

    def test_long_on_safe_loses_difficulty_awareness(self):
        # Long prediction on a non-bug → difficulty awareness should be 0.
        long_on_safe = _block(
            "long",
            "x" * 320,
            "skip_file",
            "include/types.h",
        )
        r = compute_metacognitive_reward(
            long_on_safe, bug_files={"drivers/foo.c"}  # bugs elsewhere
        )
        self.assertAlmostEqual(r.calibration, 1.0, places=3)
        self.assertAlmostEqual(r.difficulty_awareness, 0.0, places=3)

    def test_short_prediction_with_long_actual_loses_calibration(self):
        # The model said `short` but actually thought 400 chars → bad calibration.
        text = _block(
            "short",
            "x" * 400,
            "flag_vulnerable",
            "drivers/foo.c",
        )
        r = compute_metacognitive_reward(text, bug_files={"drivers/foo.c"})
        self.assertLess(r.calibration, 0.5)


class CouplingContract(unittest.TestCase):
    """Orphan predictions (no following tool call) must be punished."""

    def test_orphan_prediction_caps_metacog(self):
        # Prediction + think but NO tool call → coupling drops, raw score
        # halves due to the multiplicative term.
        orphan = (
            "<budget_prediction>long</budget_prediction>\n"
            "<think>" + "x" * 320 + "</think>\n"
        )
        r = compute_metacognitive_reward(orphan, bug_files={"drivers/foo.c"})
        self.assertEqual(r.n_predictions, 1)
        # No coupled triple → coupling is 0
        self.assertAlmostEqual(r.coupling, 0.0, places=3)
        # Multiplier is (0.5 + 0.5 * coupling) = 0.5 → raw is at most half
        self.assertLessEqual(r.raw_score, 0.5 + 1e-6)

    def test_no_predictions_returns_zero(self):
        plain = "<think>some reasoning</think>"
        r = compute_metacognitive_reward(plain, bug_files=set())
        self.assertEqual(r.n_predictions, 0)
        self.assertEqual(r.raw_score, 0.0)


class RedTeamContract(unittest.TestCase):
    """The honest policy must dominate naive reward-hack attempts."""

    def test_long_on_safe_attack_underperforms_honest(self):
        # Attacker always predicts `long` and pads — hoping calibration +
        # padded thinking + a tool-call wrap dominates the honest short
        # prediction.  Difficulty-awareness component must catch this.
        attack = _block(
            "long",
            "x" * 320,
            "skip_file",
            "include/types.h",  # safe file
        )
        honest = HONEST_SHORT_SAFE
        bugs = {"drivers/foo.c"}  # bugs exist elsewhere in the episode
        a = compute_metacognitive_reward(attack, bug_files=bugs).raw_score
        h = compute_metacognitive_reward(honest, bug_files=bugs).raw_score
        self.assertGreater(
            h, a, "honest short-on-safe must beat long-padding-on-safe attack"
        )

    def test_orphan_attack_underperforms_honest(self):
        # Predict-only attack with no tool calls.
        orphan = (
            "<budget_prediction>long</budget_prediction>\n"
            "<think>" + "x" * 320 + "</think>\n"
        ) * 3  # three orphans
        bugs = {"drivers/foo.c"}
        a = compute_metacognitive_reward(orphan, bug_files=bugs).raw_score
        h = compute_metacognitive_reward(HONEST_LONG_BUG, bug_files=bugs).raw_score
        self.assertGreater(h, a, "honest policy must beat the orphan attack")


class SystemPromptContract(unittest.TestCase):
    """The system-prompt addendum must surface the same band thresholds the
    reward uses, otherwise we have a documentation/code drift bug."""

    def test_band_thresholds_appear_in_addendum(self):
        addendum = METACOG_SYSTEM_PROMPT_ADDENDUM
        self.assertIn("0–80", addendum)
        self.assertIn("80–250", addendum)
        self.assertIn("250+", addendum)
        # Band ranges in code must match what the prompt advertises
        self.assertEqual(BAND_RANGES["short"], (0, 80))
        self.assertEqual(BAND_RANGES["medium"], (80, 250))
        self.assertEqual(BAND_RANGES["long"][0], 250)


if __name__ == "__main__":
    unittest.main()
