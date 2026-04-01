"""Prompt templates and response parsing for the LLM scanner."""

from __future__ import annotations

import re

# Matches leading numbering like "1. ", "2) ", "[1] " that LLMs sometimes add.
# Requires whitespace after punctuation to avoid false matches on bib_keys like "2023.acl-long.1".
_LEADING_BULLET_RE = re.compile(r"^(?:\d+[.)]\s+|\[\d+\]\s*)")

SYSTEM_PROMPT = """\
You are a research assistant identifying related work for an academic paper.
You will receive a proposed abstract and a batch of candidate papers (each with bib_key, title, abstract).
For each candidate that is related to the proposed abstract, output its bib_key and a brief reason.

Rules:
- Only include papers genuinely related to the proposed abstract.
- "Related" means: addresses a similar problem, uses similar methods, provides relevant background, or presents comparable results.
- Reason must be under 20 words. Use key phrases separated by semicolons. Not full sentences.
- Output format: one line per related paper, exactly: bib_key | reason
- If no papers are related, output exactly: NONE
- Do not output anything else."""


def format_batch_prompt(user_abstract: str, papers: list[dict]) -> list[dict]:
    """Format a batch of papers into chat messages for the LLM.

    Returns:
        List of message dicts [{role, content}, ...] with system + user messages.
    """
    paper_blocks = []
    for i, p in enumerate(papers, 1):
        # Truncate very long abstracts to ~300 words
        abstract = p.get("abstract", "")
        words = abstract.split()
        if len(words) > 300:
            abstract = " ".join(words[:300]) + "..."
        paper_blocks.append(
            f"[{i}] bib_key: {p['bib_key']}\nTitle: {p['title']}\nAbstract: {abstract}"
        )

    user_content = (
        "=== PROPOSED ABSTRACT ===\n"
        f"{user_abstract}\n\n"
        "=== CANDIDATE PAPERS ===\n"
        + "\n\n".join(paper_blocks)
        + "\n\n=== OUTPUT ===\n"
        "List related papers below (bib_key | reason), or NONE if none are related:"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def parse_scan_response(response_text: str, valid_keys: set[str]) -> list[dict]:
    """Parse LLM response into list of {bib_key, reason}.

    Args:
        response_text: Raw text from the LLM.
        valid_keys: Set of bib_keys that were in the batch (for validation).

    Returns:
        List of dicts with 'bib_key' and 'reason' fields.
        Only includes keys that exist in valid_keys.
    """
    results = []
    text = response_text.strip()
    if text == "NONE":
        return results

    for line in text.splitlines():
        line = line.strip()
        if not line or line == "NONE":
            continue
        if "|" not in line:
            continue

        parts = line.split("|", 1)
        bib_key = parts[0].strip()
        reason = parts[1].strip() if len(parts) > 1 else ""

        # Strip leading numbering/bullets that LLMs sometimes add (e.g. "1. ", "2) ")
        bib_key = _LEADING_BULLET_RE.sub("", bib_key)

        if bib_key in valid_keys:
            results.append({"bib_key": bib_key, "reason": reason})

    return results
