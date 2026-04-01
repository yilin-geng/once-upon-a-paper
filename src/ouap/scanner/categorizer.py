"""Second-pass categorization of matched papers into actionable categories."""

from __future__ import annotations

import re
import sqlite3

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

from ouap.config import CATEGORIES, DEFAULT_CATEGORIZE_BATCH_SIZE
from ouap.data.store import get_papers_by_keys
from ouap.scanner.providers import LLMProvider

_LEADING_BULLET_RE = re.compile(r"^(?:\d+[.)]\s+|\[\d+\]\s*)")

SYSTEM_PROMPT_CATEGORIZE = """\
You are a research advisor categorizing related papers for an academic project.
You will receive a proposed abstract and a batch of papers that have already been
identified as related. For each paper, assign exactly one category and a relevance
score.

CATEGORIES (assign exactly one per paper):

OVERLAP — The paper addresses the same research question using a similar approach.
  This is the highest-threat category: potential scooping or direct competition.
  Both the problem AND the method must overlap substantially.

SAME_PROBLEM — The paper targets the same problem or phenomenon but uses a
  meaningfully different technique, model, or framework. Good for comparison
  and "unlike X, we do Y" positioning.

SAME_METHOD — The paper uses a similar technique or methodology but applies it
  to a different problem domain. Validates the technical approach. Does NOT
  overlap on the research question.

FOUNDATION — Seminal work, survey, or widely-known paper that establishes
  concepts, benchmarks, or datasets that the proposed work builds upon.
  Must be genuinely foundational to this specific research area.

SUPPORT — The paper provides findings or evidence that strengthen the motivation
  or plausibility of the proposed approach. The paper does NOT need to address
  the same problem — it just provides corroborating evidence.

CONTRA — The paper presents results, arguments, or evidence that challenges the
  proposed approach. This includes: negative results for similar methods,
  evidence that undermines assumptions, or work showing the proposed problem
  is already solved or unimportant. Be vigilant for this category.

TANGENTIAL — Related to the broad topic but does not directly engage with the
  specific problem, method, or findings. Peripheral connection only.

RELEVANCE SCORE (1-5):
5 = Critical: must read and cite
4 = Important: should read and likely cite
3 = Relevant: useful context, worth citing
2 = Minor: loose connection, cite if space permits
1 = Marginal: very weak connection

DISAMBIGUATION RULES — apply in this order:
1. If the paper's findings challenge your approach or assumptions → CONTRA
2. If BOTH the problem and method overlap substantially → OVERLAP
3. If only the problem overlaps (different method) → SAME_PROBLEM
4. If only the method overlaps (different problem) → SAME_METHOD
5. If the paper defines foundational concepts, benchmarks, or datasets you build on → FOUNDATION
6. If the paper's findings corroborate your motivation → SUPPORT
7. Otherwise → TANGENTIAL

OUTPUT FORMAT: one line per paper, exactly:
bib_key | CATEGORY | score | reason

- Reason must be under 20 words. Key phrases separated by semicolons.
- Output every paper in the batch. Do not skip any.
- Do not output anything else."""


def format_categorize_prompt(
    user_abstract: str, papers: list[dict]
) -> list[dict]:
    """Format a batch of already-matched papers for categorization.

    Each paper dict must have: bib_key, title, abstract, scan_reason.
    """
    paper_blocks = []
    for i, p in enumerate(papers, 1):
        abstract = p.get("abstract", "")
        words = abstract.split()
        if len(words) > 300:
            abstract = " ".join(words[:300]) + "..."
        block = (
            f"[{i}] bib_key: {p['bib_key']}\n"
            f"Title: {p['title']}\n"
            f"Abstract: {abstract}\n"
            f"Initial scan reason: {p.get('scan_reason', '')}"
        )
        paper_blocks.append(block)

    user_content = (
        "=== PROPOSED ABSTRACT ===\n"
        f"{user_abstract}\n\n"
        "=== PAPERS TO CATEGORIZE ===\n"
        + "\n\n".join(paper_blocks)
        + "\n\n=== OUTPUT ===\n"
        "Categorize every paper above (bib_key | CATEGORY | score | reason):"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT_CATEGORIZE},
        {"role": "user", "content": user_content},
    ]


def parse_categorize_response(
    response_text: str, valid_keys: set[str]
) -> list[dict]:
    """Parse categorization response into list of {bib_key, category, relevance_score, reason}.

    Falls back to TANGENTIAL/2 for parse failures.
    """
    valid_categories = set(CATEGORIES)
    results = []

    for line in response_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split("|")
        if len(parts) < 4:
            continue

        bib_key = parts[0].strip()
        category = parts[1].strip().upper()
        score_str = parts[2].strip()
        reason = parts[3].strip()

        # Strip leading numbering
        bib_key = _LEADING_BULLET_RE.sub("", bib_key)

        if bib_key not in valid_keys:
            continue

        # Validate category
        if category not in valid_categories:
            category = "TANGENTIAL"

        # Validate score
        try:
            score = int(score_str)
            score = max(1, min(5, score))
        except ValueError:
            score = 2

        results.append({
            "bib_key": bib_key,
            "category": category,
            "relevance_score": score,
            "reason": reason,
        })

    return results


def categorize(
    conn: sqlite3.Connection,
    user_abstract: str,
    matches: list[dict],
    provider: LLMProvider,
    batch_size: int = DEFAULT_CATEGORIZE_BATCH_SIZE,
) -> list[dict]:
    """Categorize already-matched papers.

    Args:
        conn: Database connection.
        user_abstract: The proposed abstract.
        matches: List of {bib_key, reason} from the scan phase.
        provider: LLM provider instance.
        batch_size: Papers per categorization batch.

    Returns:
        List of {bib_key, category, relevance_score, reason} for all papers.
        Papers that fail categorization get TANGENTIAL/2.
    """
    if not matches:
        return []

    # Fetch full paper data for matched papers
    bib_keys = [m["bib_key"] for m in matches]
    papers = get_papers_by_keys(conn, bib_keys)
    paper_map = {p["bib_key"]: p for p in papers}

    # Build enriched paper list with scan reasons
    enriched = []
    for m in matches:
        paper = paper_map.get(m["bib_key"], {})
        enriched.append({
            "bib_key": m["bib_key"],
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract", ""),
            "scan_reason": m.get("reason", ""),
        })

    batches = [enriched[i : i + batch_size] for i in range(0, len(enriched), batch_size)]
    categorized = {}

    with Progress(
        TextColumn("[bold magenta]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(
            f"Categorizing {len(enriched)} papers ({len(batches)} batches)...",
            total=len(batches),
        )

        for batch in batches:
            messages = format_categorize_prompt(user_abstract, batch)
            response = provider.complete(messages)

            valid_keys = {p["bib_key"] for p in batch}
            parsed = parse_categorize_response(response, valid_keys)

            for item in parsed:
                categorized[item["bib_key"]] = item

            progress.advance(task)

    # Build final results, with fallback for uncategorized papers
    results = []
    for m in matches:
        key = m["bib_key"]
        if key in categorized:
            results.append(categorized[key])
        else:
            results.append({
                "bib_key": key,
                "category": "TANGENTIAL",
                "relevance_score": 2,
                "reason": m.get("reason", ""),
            })

    return results
