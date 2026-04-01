"""Core scanning engine: batching, LLM dispatch, result collection."""

from __future__ import annotations

import sqlite3

from rich.progress import Progress, BarColumn, MofNCompleteColumn, TextColumn, TimeRemainingColumn

from ouap.config import DEFAULT_BATCH_SIZE
from ouap.data.store import query_papers
from ouap.scanner.prompt import format_batch_prompt, parse_scan_response
from ouap.scanner.providers import LLMProvider


def scan(
    conn: sqlite3.Connection,
    user_abstract: str,
    venues: list[str] | None,
    year_min: int,
    year_max: int,
    provider: LLMProvider,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[dict]:
    """Run the full scanning loop.

    Args:
        conn: Database connection.
        user_abstract: The proposed abstract to match against.
        venues: Venue filter (None = all).
        year_min/year_max: Year range.
        provider: LLM provider instance.
        batch_size: Papers per LLM call.

    Returns:
        List of {bib_key, reason} for all matched papers.
    """
    papers = query_papers(conn, venues, year_min, year_max)
    if not papers:
        return []

    batches = [papers[i : i + batch_size] for i in range(0, len(papers), batch_size)]
    all_matches = []

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(
            f"Scanning {len(papers)} papers ({len(batches)} batches)...",
            total=len(batches),
        )

        for batch in batches:
            messages = format_batch_prompt(user_abstract, batch)
            response = provider.complete(messages)

            valid_keys = {p["bib_key"] for p in batch}
            matches = parse_scan_response(response, valid_keys)
            all_matches.extend(matches)

            progress.advance(task)

    return all_matches
