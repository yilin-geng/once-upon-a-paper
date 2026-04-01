"""Output generation: .bib file assembly and .json report."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ouap.config import CATEGORY_DISPLAY_ORDER, CATEGORY_LABELS
from ouap.data.store import get_papers_by_keys


def write_bib_file(
    conn: sqlite3.Connection, matches: list[dict], output_path: Path
) -> int:
    """Write a .bib file containing only matched papers.

    If matches have 'category' and 'relevance_score' fields, adds a comment
    before each entry.

    Returns:
        Number of papers written.
    """
    bib_keys = [m["bib_key"] for m in matches]
    papers = get_papers_by_keys(conn, bib_keys)
    paper_map = {p["bib_key"]: p for p in papers}

    # If categorized, sort by category display order then by score descending
    has_categories = matches and "category" in matches[0]
    if has_categories:
        order = {c: i for i, c in enumerate(CATEGORY_DISPLAY_ORDER)}
        sorted_matches = sorted(
            matches,
            key=lambda m: (order.get(m.get("category", "TANGENTIAL"), 99), -m.get("relevance_score", 0)),
        )
    else:
        sorted_matches = matches

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for m in sorted_matches:
            paper = paper_map.get(m["bib_key"])
            if paper and paper.get("bibtex"):
                if has_categories:
                    cat = m.get("category", "TANGENTIAL")
                    score = m.get("relevance_score", 0)
                    label = CATEGORY_LABELS.get(cat, cat)
                    f.write(f"% Category: {label} (relevance {score}/5)\n")
                    f.write(f"% Reason: {m.get('reason', '')}\n")
                f.write(paper["bibtex"])
                f.write("\n\n")
                count += 1
    return count


def write_json_report(
    conn: sqlite3.Connection, matches: list[dict], output_path: Path
) -> int:
    """Write a JSON report.

    If matches have 'category'/'relevance_score', produces a grouped report
    with summary statistics. Otherwise produces the flat format.

    Returns:
        Number of entries written.
    """
    bib_keys = [m["bib_key"] for m in matches]
    papers = get_papers_by_keys(conn, bib_keys)
    paper_map = {p["bib_key"]: p for p in papers}

    has_categories = matches and "category" in matches[0]

    if has_categories:
        report = _build_categorized_report(matches, paper_map)
    else:
        report = _build_flat_report(matches, paper_map)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return len(matches)


def _build_flat_report(matches: list[dict], paper_map: dict) -> dict:
    """Original flat format: {bib_key: {reason, title, venue, year}}."""
    report = {}
    for m in matches:
        key = m["bib_key"]
        paper = paper_map.get(key, {})
        report[key] = {
            "reason": m["reason"],
            "title": paper.get("title", ""),
            "venue": paper.get("venue", ""),
            "year": paper.get("year"),
        }
    return report


def _build_categorized_report(matches: list[dict], paper_map: dict) -> dict:
    """Categorized format with summary and grouped papers."""
    # Build summary
    cat_counts: dict[str, int] = {}
    cat_scores: dict[str, list[int]] = {}
    for m in matches:
        cat = m.get("category", "TANGENTIAL")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        cat_scores.setdefault(cat, []).append(m.get("relevance_score", 0))

    summary = {
        "total_papers": len(matches),
        "categories": {},
    }
    for cat in CATEGORY_DISPLAY_ORDER:
        if cat in cat_counts:
            scores = cat_scores[cat]
            summary["categories"][cat] = {
                "label": CATEGORY_LABELS.get(cat, cat),
                "count": cat_counts[cat],
                "avg_relevance": round(sum(scores) / len(scores), 1),
            }

    # Build papers dict
    papers = {}
    for m in matches:
        key = m["bib_key"]
        paper = paper_map.get(key, {})
        papers[key] = {
            "category": m.get("category", "TANGENTIAL"),
            "relevance_score": m.get("relevance_score", 0),
            "reason": m.get("reason", ""),
            "title": paper.get("title", ""),
            "venue": paper.get("venue", ""),
            "year": paper.get("year"),
        }

    return {"summary": summary, "papers": papers}


def write_report_file(report_markdown: str, output_path: Path) -> None:
    """Write the human-readable report markdown to disk."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_markdown)
