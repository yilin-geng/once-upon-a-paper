"""Phase 3: Generate a human-readable related-work report via per-section LLM calls."""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

from ouap.config import (
    CATEGORY_DISPLAY_ORDER,
    CATEGORY_LABELS,
    DEFAULT_REPORT_MAX_PAPERS_PER_SECTION,
)
from ouap.data.store import get_papers_by_keys
from ouap.scanner.providers import LLMProvider

# ---------------------------------------------------------------------------
# Shared style preamble injected into every section prompt
# ---------------------------------------------------------------------------

_STYLE_PREAMBLE = """\
You are a direct research advisor writing a concise report section for a colleague.

Rules:
- Cite papers as \\cite{bib_key} so the text is copy-pasteable into LaTeX.
- Use short paragraphs (2-4 sentences). No bullet-point walls.
- Do not restate the user's abstract; the reader knows their own paper.
- No filler phrases ("It is worth noting", "In summary", "Overall").
- Every sentence must convey information beyond what paper titles alone say.
- Do not add section headings — the report framework supplies them."""

# ---------------------------------------------------------------------------
# Per-section system prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_NOVELTY = _STYLE_PREAMBLE + """

TASK: Write the "Novelty and Risk Assessment" section.

You receive the user's proposed abstract and papers categorized as OVERLAP
(same problem AND similar method) and CONTRA (evidence challenging the approach).

For each OVERLAP paper: state what overlaps (problem + method), what differs,
and whether it constitutes a genuine scooping risk or can be differentiated.

For each CONTRA paper: state what finding challenges the user's approach and
what the user should address in their paper.

If a category is empty, state that in one sentence (e.g., "No direct overlap
was found in the scanned venues.") and move on.

End with 2-3 sentences of positioning advice: how the author should frame
their contribution given these risks."""

SYSTEM_PROMPT_LANDSCAPE = _STYLE_PREAMBLE + """

TASK: Write the "Competitive Landscape" section.

You receive papers that target the same problem but use different methods
(SAME_PROBLEM category).

Do NOT list papers one by one. Cluster them by approach or theme. Name each
cluster (e.g., "Training-based personality shaping", "Prompt-level role
assignment"). For each cluster: cite the constituent papers, explain how the
approach family differs from the user's method, and note which papers are the
strongest comparisons or natural baselines.

If there are too many clusters, prioritize the 3-4 most populated or highest-
relevance ones and mention the rest briefly."""

SYSTEM_PROMPT_METHOD = _STYLE_PREAMBLE + """

TASK: Write the "Methodological Context" section.

You receive two groups of papers:
- SAME_METHOD: papers using a similar technique on a different problem.
- FOUNDATION: seminal work, surveys, benchmarks the proposed work builds on.

Organize by technical concept (e.g., "activation steering", "probe-based
analysis"), not by paper. For SAME_METHOD papers: note where the method has
been validated and what limitations were found in those applications. For
FOUNDATION papers: mention each briefly — the reader likely knows them.

State how this technical lineage supports or informs the user's choices."""

SYSTEM_PROMPT_SUPPORT = _STYLE_PREAMBLE + """

TASK: Write the "Supporting Evidence and Gaps" section.

You receive SUPPORT papers (findings that corroborate the user's motivation)
plus summary statistics of all categories.

First, synthesize the supporting evidence. Group by which aspect of the
argument each paper bolsters.

Then, comment on gaps visible in the landscape: categories with zero or very
few papers, under-explored angles, or potential weaknesses in the literature
coverage. Keep gap analysis to 2-3 sentences — do not speculate extensively."""

SYSTEM_PROMPT_READING = _STYLE_PREAMBLE + """

TASK: Write the "Recommended Reading Order" section.

You receive all matched papers with their category, relevance score, and reason.

Produce a numbered list organized into three tiers:
- **Read immediately** (5-7 papers): OVERLAP, CONTRA, and high-relevance
  SAME_PROBLEM papers (score 4-5).
- **Read soon** (5-7 papers): remaining SAME_PROBLEM, SAME_METHOD, and
  FOUNDATION papers with score 3+.
- **Skim if time permits** (3-5 papers): SUPPORT papers with score 3+ and
  any other notable papers.

Each entry: one line.
Format: `N. \\cite{bib_key} — *Title* — reason in 10 words or fewer`

Do not include TANGENTIAL papers unless they scored 4+.
Do not include any paper with relevance score 1."""

# ---------------------------------------------------------------------------
# Section definitions
# ---------------------------------------------------------------------------

REPORT_SECTIONS = [
    {
        "id": 1,
        "title": "Novelty and Risk Assessment",
        "system_prompt": SYSTEM_PROMPT_NOVELTY,
        "categories": ["OVERLAP", "CONTRA"],
        "full_abstracts": True,
    },
    {
        "id": 2,
        "title": "Competitive Landscape",
        "system_prompt": SYSTEM_PROMPT_LANDSCAPE,
        "categories": ["SAME_PROBLEM"],
        "full_abstracts": True,
    },
    {
        "id": 3,
        "title": "Methodological Context",
        "system_prompt": SYSTEM_PROMPT_METHOD,
        "categories": ["SAME_METHOD", "FOUNDATION"],
        "full_abstracts": True,
    },
    {
        "id": 4,
        "title": "Supporting Evidence and Gaps",
        "system_prompt": SYSTEM_PROMPT_SUPPORT,
        "categories": ["SUPPORT"],
        "full_abstracts": True,
    },
    {
        "id": 5,
        "title": "Recommended Reading Order",
        "system_prompt": SYSTEM_PROMPT_READING,
        "categories": None,  # all papers
        "full_abstracts": False,
    },
]

# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def _format_paper_block(paper: dict, match: dict, include_abstract: bool) -> str:
    """Format a single paper for inclusion in a prompt."""
    lines = [
        f"bib_key: {match['bib_key']}",
        f"Title: {paper.get('title', '')}",
        f"Category: {match.get('category', '')} | Relevance: {match.get('relevance_score', '')}/5",
        f"Reason: {match.get('reason', '')}",
    ]
    if include_abstract:
        abstract = paper.get("abstract", "") or ""
        words = abstract.split()
        if len(words) > 300:
            abstract = " ".join(words[:300]) + "..."
        lines.append(f"Abstract: {abstract}")
    return "\n".join(lines)


def _format_section_prompt(
    section: dict,
    user_abstract: str,
    papers_for_section: list[tuple[dict, dict]],  # (paper_row, match_dict)
    summary_stats: dict | None,
) -> list[dict]:
    """Build the message list for one report section."""
    include_abstract = section["full_abstracts"]

    # Sort by relevance descending; cap full-abstract papers
    sorted_papers = sorted(
        papers_for_section,
        key=lambda pm: -pm[1].get("relevance_score", 0),
    )

    max_full = DEFAULT_REPORT_MAX_PAPERS_PER_SECTION
    blocks = []
    for i, (paper, match) in enumerate(sorted_papers):
        use_abstract = include_abstract and i < max_full
        blocks.append(_format_paper_block(paper, match, use_abstract))

    user_parts = [
        "=== PROPOSED ABSTRACT ===",
        user_abstract,
        "",
        f"=== PAPERS ({len(sorted_papers)} total) ===",
        "\n\n".join(blocks),
    ]

    # Section 4 also gets summary stats
    if summary_stats and section["id"] == 4:
        stat_lines = ["", "=== LANDSCAPE SUMMARY ==="]
        for cat in CATEGORY_DISPLAY_ORDER:
            info = summary_stats.get(cat)
            if info:
                stat_lines.append(
                    f"{CATEGORY_LABELS.get(cat, cat)}: {info['count']} papers, "
                    f"avg relevance {info['avg_relevance']}"
                )
            else:
                stat_lines.append(f"{CATEGORY_LABELS.get(cat, cat)}: 0 papers")
        user_parts.extend(stat_lines)

    return [
        {"role": "system", "content": section["system_prompt"]},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _build_summary_stats(categorized: list[dict]) -> dict:
    """Build per-category summary statistics (mirroring output._build_categorized_report)."""
    cat_counts: dict[str, int] = {}
    cat_scores: dict[str, list[int]] = {}
    for m in categorized:
        cat = m.get("category", "TANGENTIAL")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        cat_scores.setdefault(cat, []).append(m.get("relevance_score", 0))

    stats = {}
    for cat in CATEGORY_DISPLAY_ORDER:
        if cat in cat_counts:
            scores = cat_scores[cat]
            stats[cat] = {
                "count": cat_counts[cat],
                "avg_relevance": round(sum(scores) / len(scores), 1),
            }
    return stats


def generate_report(
    conn: sqlite3.Connection,
    user_abstract: str,
    categorized: list[dict],
    provider: LLMProvider,
    venues_scanned: list[str] | None,
    year_min: int,
    year_max: int,
) -> str:
    """Generate a multi-section prose report from categorized matches.

    Args:
        conn: Database connection.
        user_abstract: The proposed abstract.
        categorized: List of {bib_key, category, relevance_score, reason}.
        provider: LLM provider instance.
        venues_scanned: Venue list (None = all).
        year_min, year_max: Year range scanned.

    Returns:
        Complete markdown report as a string.
    """
    # Fetch full paper data
    bib_keys = [m["bib_key"] for m in categorized]
    papers = get_papers_by_keys(conn, bib_keys)
    paper_map = {p["bib_key"]: p for p in papers}

    # Group matches by category
    by_category: dict[str, list[dict]] = {}
    for m in categorized:
        by_category.setdefault(m.get("category", "TANGENTIAL"), []).append(m)

    summary_stats = _build_summary_stats(categorized)

    # Generate each section
    section_texts: list[str] = []

    with Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(
            f"Generating report (5 sections)...",
            total=len(REPORT_SECTIONS),
        )

        for section in REPORT_SECTIONS:
            # Collect papers for this section
            if section["categories"] is None:
                # Section 5: all papers
                section_papers = [
                    (paper_map.get(m["bib_key"], {}), m) for m in categorized
                ]
            else:
                section_papers = []
                for cat in section["categories"]:
                    for m in by_category.get(cat, []):
                        paper = paper_map.get(m["bib_key"], {})
                        section_papers.append((paper, m))

            # Skip LLM call if section has zero papers (except section 4 which also does gap analysis)
            if not section_papers and section["id"] != 4:
                section_texts.append(
                    f"No papers in this category were found in the scanned venues."
                )
                progress.advance(task)
                continue

            messages = _format_section_prompt(
                section, user_abstract, section_papers, summary_stats
            )
            text = provider.complete(messages)
            section_texts.append(text.strip())
            progress.advance(task)

    # Assemble markdown
    venue_str = ", ".join(venues_scanned) if venues_scanned else "all venues"
    header = (
        f"# Related Work Report\n\n"
        f"**Generated by ouap** | Model: {provider.model_name} | "
        f"Date: {date.today().isoformat()}\n"
        f"**Scanned:** {len(categorized)} papers across {venue_str} "
        f"({year_min}\u2013{year_max})\n\n---\n"
    )

    body_parts = []
    for section, text in zip(REPORT_SECTIONS, section_texts):
        body_parts.append(f"\n## {section['id']}. {section['title']}\n\n{text}\n")

    footer = (
        f"\n---\n\n*{len(categorized)} papers analyzed. "
        f"Full data: related.json | BibTeX: related.bib*\n"
    )

    return header + "\n".join(body_parts) + footer
