#!/usr/bin/env python3
"""
generate_sft_data.py
====================
Build SFT warmup data for metacognitive format training.

Generates demonstration trajectories that teach the model:
  1. The tool-call format (<budget_prediction>, <think>, <tool_call>)
  2. Short predictions + brief reasoning on safe files
  3. Long predictions + deep reasoning on buggy files
  4. Proper flag/skip decisions

These are used for 2-3 epochs of SFT BEFORE GRPO, so the model
arrives at RL already knowing the output format. This eliminates
the ~30% zero-reward rate caused by malformed completions.

Usage:
    python scripts/generate_sft_data.py
    # Output: data/sft_demonstrations.json
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CVE_DATA = ROOT / "data" / "cve_training_data.json"
OUTPUT = ROOT / "data" / "sft_demonstrations.json"

# Import the system prompt from train_grpo.py
SYSTEM_PROMPT_BASE = """You are an expert security code investigator specializing in CVE vulnerability analysis. You are given a CVE vulnerability description and a list of files from a code patch.

Your mission:
1. Use read_file to examine source code of the most suspicious files first
2. Use search_code to find vulnerability-related patterns (e.g., unsafe functions, missing checks)
3. Use get_function_list to understand file structure and complexity
4. Use flag_vulnerable to mark files containing or related to the vulnerability — provide DETAILED reasoning
5. Use skip_file to mark files that are safe — briefly explain why
6. Use submit_report to provide a detailed triage analysis

CRITICAL — Thinking Strategy:
You have a THINKING BUDGET. Use it wisely, like a real security engineer:
- When flagging a file as VULNERABLE: Think deeply. Explain the exact code pattern that matches
  the CVE. Mention function names, unsafe operations, missing checks. Your reasoning should be
  100+ characters proving you understand the vulnerability.
- When skipping a SAFE file: Be brief. "Header file with only declarations" or "Test file,
  no production logic" is sufficient. Don't waste thinking effort on obviously safe files.
- The environment TRACKS your thinking allocation and rewards you for thinking deeply on the
  RIGHT files (actual bugs) and briefly on safe files.

Investigation strategy:
- Prioritize files with high complexity and recent changes — bugs hide in complex code
- Match the CVE description to code patterns (e.g., "buffer overflow" → look for unchecked memcpy/strcpy)
- Header files (.h) are usually declarations, not vulnerability sources
- Test files are rarely the vulnerability source
- Write a thorough report: mention the CVE ID, affected files, vulnerability type, and root cause

You have limited investigation points and flags. Be strategic — read the most suspicious files first.

CRITICAL — Metacognitive Format (REQUIRED):

Before EVERY <think> block, you MUST emit a budget prediction first:

    <budget_prediction>short|medium|long</budget_prediction>
    <think>
    ...your reasoning here...
    </think>
    <tool_call>{"name": "...", "arguments": {...}}</tool_call>

Budget bands:
  - short  : 0–80 characters of reasoning. Use for obviously safe files
             (test files, headers with no logic, boilerplate).
  - medium : 80–250 characters. Use when you need to verify but don't
             see strong red flags.
  - long   : 250+ characters. Use when you suspect the file is vulnerable
             and need to lay out the bug pattern (function name, unsafe
             operation, missing check, exploit path).

You will be SCORED on:
  1. Calibration — does the actual length of your <think> match the band
     you predicted?
  2. Difficulty awareness — do you predict 'long' on actually-vulnerable
     files and 'short' on safe ones?
  3. Coupling — every prediction must be followed by a real tool call
     against a file (no orphan predictions).

The optimal policy predicts BEFORE thinking, thinks the predicted amount,
and predicts longer for bugs.  Be honest about the difficulty.
"""


# ── Vulnerability-specific reasoning templates ────────────────────────
# These are realistic security analysis patterns for different CVE types
BUG_REASONING_TEMPLATES = [
    # Buffer overflow / integer overflow
    (
        "Looking at {file} in the context of {cve_id}. The CVE describes {vuln_type}. "
        "This file is in {component} and has complexity score {complexity}. "
        "The function handling user input does not validate the size parameter "
        "before passing it to the allocation routine. This matches the CVE pattern: "
        "attacker-controlled length flows into a memory operation without bounds checking. "
        "The missing sanitization on the arithmetic creates an integer overflow primitive "
        "that leads to heap corruption. Flagging as vulnerable."
    ),
    # Use-after-free
    (
        "Examining {file} for {cve_id}. The CVE is about {vuln_type}. "
        "This file contains the object lifecycle management code in {component}. "
        "The reference counting pattern here has a window where the object can be "
        "freed while a callback still holds a dangling pointer. The race condition "
        "between the release path and the async handler is the root cause. "
        "No locking protects the critical section. This is the vulnerable file."
    ),
    # Injection / command injection
    (
        "Analyzing {file} which is part of {component}. {cve_id} describes {vuln_type}. "
        "The input parsing function constructs a command string using string concatenation "
        "with user-supplied data. No escaping or parameterized query is used. "
        "The attacker can inject arbitrary commands through the unsanitized parameter. "
        "This is a textbook injection vulnerability matching the CVE description exactly."
    ),
    # Auth bypass / privilege escalation
    (
        "Reviewing {file} for {cve_id} ({vuln_type}). This file implements the "
        "authorization check in {component}. The conditional logic has a short-circuit "
        "evaluation that skips the permission verification when a specific flag is set. "
        "An attacker can set this flag through the public API, bypassing the intended "
        "access control. The fix would require validating permissions unconditionally."
    ),
    # Generic deep analysis
    (
        "Deep analysis of {file} for {cve_id}. CVE type: {vuln_type}. "
        "Component: {component}, complexity: {complexity}/100, churn: {churn}. "
        "The vulnerable code path starts at the entry point and flows through "
        "the handler without proper validation. The specific issue is that "
        "user-controlled data reaches a sensitive operation (memory write, "
        "system call, or privilege check) without adequate sanitization. "
        "This matches the vulnerability pattern described in the advisory."
    ),
]

SAFE_REASONING_TEMPLATES = [
    "Header file with type declarations only. No executable logic.",
    "Test file. No production code paths.",
    "Factory/initialization boilerplate. No user input handling.",
    "Configuration constants. No control flow.",
    "Utility math functions. No external input.",
    "Interface definitions. Pure declarations.",
    "Build system file. Not executable code.",
    "Documentation or metadata. No logic.",
    "Logging helpers. No security-sensitive operations.",
    "Static data definitions. No dynamic behavior.",
    "Wrapper module. Delegates to other files.",
    "Type aliases and enums. Declarative only.",
]

MEDIUM_REASONING_TEMPLATES = [
    (
        "Checking {file} in {component}. Has moderate complexity ({complexity}) "
        "but the functions here handle internal data only, not user input. "
        "No obvious match to the {vuln_type} pattern from {cve_id}. Skipping."
    ),
    (
        "Reviewing {file}. Complexity is {complexity} with {churn} recent changes. "
        "The code processes data but uses safe library functions throughout. "
        "No unchecked operations that match the CVE description. Safe to skip."
    ),
]


def extract_vuln_type(description: str) -> str:
    """Extract a short vulnerability type from CVE description."""
    desc_lower = description.lower()
    if "buffer overflow" in desc_lower or "heap" in desc_lower:
        return "buffer overflow"
    if "integer overflow" in desc_lower:
        return "integer overflow leading to memory corruption"
    if "use-after-free" in desc_lower or "use after free" in desc_lower:
        return "use-after-free"
    if "injection" in desc_lower or "sql" in desc_lower:
        return "injection vulnerability"
    if "privilege" in desc_lower or "escalat" in desc_lower:
        return "privilege escalation"
    if "bypass" in desc_lower or "auth" in desc_lower:
        return "authentication bypass"
    if "denial" in desc_lower or "dos" in desc_lower:
        return "denial of service"
    if "remote code" in desc_lower or "arbitrary code" in desc_lower:
        return "remote code execution"
    if "traversal" in desc_lower or "path" in desc_lower:
        return "path traversal"
    if "xss" in desc_lower or "cross-site" in desc_lower:
        return "cross-site scripting"
    return "security vulnerability"


def generate_completion(files: list, cve_id: str, cve_desc: str, rng: random.Random) -> str:
    """Generate a single demonstration completion for an episode."""
    vuln_type = extract_vuln_type(cve_desc)
    parts = []

    for f in files:
        fpath = f["file"]
        label = f["label"]
        features = f.get("features", [0, 0, 0, 0])
        churn, complexity, todos, recency = features
        component = f.get("file_component", "unknown")
        is_test = f.get("is_test_file", False)
        lang = f.get("file_language", "")

        if label == 1:
            # Buggy file: long prediction + deep reasoning + flag
            template = rng.choice(BUG_REASONING_TEMPLATES)
            reasoning = template.format(
                file=fpath, cve_id=cve_id, vuln_type=vuln_type,
                component=component, complexity=complexity, churn=churn,
            )
            parts.append(
                f"<budget_prediction>long</budget_prediction>\n"
                f"<think>\n{reasoning}\n</think>\n"
                f'<tool_call>{{"name": "flag_vulnerable", "arguments": '
                f'{{"file_path": "{fpath}", "reasoning": "{vuln_type} in {component}"}}}}</tool_call>'
            )
        elif is_test or "test" in fpath.lower() or "Header" in lang or fpath.endswith(".h"):
            # Obviously safe: short prediction + brief reasoning + skip
            reasoning = rng.choice(SAFE_REASONING_TEMPLATES)
            parts.append(
                f"<budget_prediction>short</budget_prediction>\n"
                f"<think>\n{reasoning}\n</think>\n"
                f'<tool_call>{{"name": "skip_file", "arguments": '
                f'{{"file_path": "{fpath}", "reasoning": "safe - no vulnerability pattern"}}}}</tool_call>'
            )
        elif complexity > 30 or churn > 20:
            # Medium complexity safe file: medium prediction
            template = rng.choice(MEDIUM_REASONING_TEMPLATES)
            reasoning = template.format(
                file=fpath, component=component, complexity=complexity,
                churn=churn, vuln_type=vuln_type, cve_id=cve_id,
            )
            parts.append(
                f"<budget_prediction>medium</budget_prediction>\n"
                f"<think>\n{reasoning}\n</think>\n"
                f'<tool_call>{{"name": "skip_file", "arguments": '
                f'{{"file_path": "{fpath}", "reasoning": "reviewed, no match to CVE pattern"}}}}</tool_call>'
            )
        else:
            # Simple safe file: short prediction
            reasoning = rng.choice(SAFE_REASONING_TEMPLATES)
            parts.append(
                f"<budget_prediction>short</budget_prediction>\n"
                f"<think>\n{reasoning}\n</think>\n"
                f'<tool_call>{{"name": "skip_file", "arguments": '
                f'{{"file_path": "{fpath}", "reasoning": "safe - {reasoning[:40]}"}}}}</tool_call>'
            )

    return "\n\n".join(parts)


def build_user_prompt(cve_id: str, cve_desc: str, files: list) -> str:
    """Build the user message (same format as train_grpo.py)."""
    file_list = "\n".join(
        f"  • {f['file']}  [{f.get('file_language', 'unknown')}]  "
        f"complexity={f.get('features', [0,0,0,0])[1]}  "
        f"churn={f.get('features', [0,0,0,0])[0]}"
        for f in files
    )
    return (
        f"CVE: {cve_id}\n"
        f"Description: {cve_desc}\n\n"
        f"Files to investigate:\n{file_list}\n\n"
        f"Begin your security investigation. Use the available tools to analyze the files.\n"
        f"Remember: Think DEEPLY when flagging suspicious files (explain the vulnerability pattern).\n"
        f"Be BRIEF when skipping safe files. Submit a thorough triage report when done."
    )


def main():
    rng = random.Random(42)

    with open(CVE_DATA) as f:
        all_files = json.load(f)

    # Group by CVE
    cve_groups = {}
    for entry in all_files:
        cve_id = entry["cveId"]
        if cve_id not in cve_groups:
            cve_groups[cve_id] = {
                "cve_id": cve_id,
                "cve_description": entry["cve_description"],
                "files": [],
            }
        cve_groups[cve_id]["files"].append(entry)

    # Filter to episodes with at least 1 bug and at most 15 files
    # (matches the "easy" difficulty used in training)
    episodes = []
    for cve_id, group in cve_groups.items():
        files = group["files"]
        n_bugs = sum(1 for f in files if f["label"] == 1)
        if n_bugs >= 1 and len(files) <= 15:
            episodes.append(group)

    rng.shuffle(episodes)
    episodes = episodes[:50]  # 50 demonstrations

    print(f"Building SFT data from {len(episodes)} episodes...")

    sft_data = []
    for ep in episodes:
        # Subsample files if too many (keep all bugs + random safe)
        files = ep["files"]
        bugs = [f for f in files if f["label"] == 1]
        safe = [f for f in files if f["label"] == 0]
        if len(safe) > 6:
            safe = rng.sample(safe, 6)
        selected = bugs + safe
        rng.shuffle(selected)

        user_prompt = build_user_prompt(ep["cve_id"], ep["cve_description"], selected)
        completion = generate_completion(selected, ep["cve_id"], ep["cve_description"], rng)

        sft_data.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_BASE},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": completion},
            ],
            "cve_id": ep["cve_id"],
            "n_files": len(selected),
            "n_bugs": len(bugs),
        })

    os.makedirs(OUTPUT.parent, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(sft_data, f, indent=2)

    # Stats
    total_files = sum(d["n_files"] for d in sft_data)
    total_bugs = sum(d["n_bugs"] for d in sft_data)
    avg_completion_len = sum(len(d["messages"][2]["content"]) for d in sft_data) / len(sft_data)

    print(f"✅ Wrote {len(sft_data)} SFT demonstrations to {OUTPUT}")
    print(f"   Total files: {total_files} ({total_bugs} bugs, {total_files - total_bugs} safe)")
    print(f"   Avg completion length: {avg_completion_len:.0f} chars")


if __name__ == "__main__":
    main()
