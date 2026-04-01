"""Semantic Scholar bulk search API fetcher."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import requests
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from ouap.config import S2_BULK_URL, S2_FIELDS


def _escape_bibtex(text: str) -> str:
    """Escape special BibTeX characters in text."""
    # Braces are used as delimiters — escape & and % which are BibTeX specials
    return text.replace("&", r"\&").replace("%", r"\%")


def _make_bibtex(paper: dict, venue: str) -> str:
    """Construct a BibTeX @inproceedings entry from Semantic Scholar paper data."""
    pid = paper["paperId"]
    bib_key = f"s2-{pid[:12]}"
    authors_list = paper.get("authors") or []
    authors_str = " and ".join(a.get("name", "") for a in authors_list)
    title = _escape_bibtex(paper.get("title", ""))
    year = paper.get("year", "")
    abstract = _escape_bibtex(paper.get("abstract", ""))

    # Get DOI if available
    ext_ids = paper.get("externalIds") or {}
    doi = ext_ids.get("DOI", "")

    lines = [
        f"@inproceedings{{{bib_key},",
        f"  title = {{{title}}},",
        f"  author = {{{authors_str}}},",
        f"  year = {{{year}}},",
        f"  booktitle = {{{venue}}},",
    ]
    if abstract:
        lines.append(f"  abstract = {{{abstract}}},")
    if doi:
        lines.append(f"  doi = {{{doi}}},")
    lines.append("}")
    return "\n".join(lines)


def fetch_s2_papers(
    venue: str,
    year_min: int = 2016,
    year_max: int = 2026,
    api_key: str | None = None,
) -> list[dict]:
    """Fetch all papers for a venue/year range from Semantic Scholar bulk search.

    Args:
        venue: Venue name as recognized by Semantic Scholar (e.g., "NeurIPS").
        year_min/year_max: Year range filter.
        api_key: Optional S2 API key for higher rate limits.

    Returns:
        List of dicts ready for store.upsert_papers().
    """
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    params: dict = {
        "venue": venue,
        "year": f"{year_min}-{year_max}",
        "fields": S2_FIELDS,
    }

    now = datetime.now(timezone.utc).isoformat()
    papers = []
    token = None
    page = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
    ) as progress:
        task = progress.add_task(f"Fetching {venue} from Semantic Scholar...", total=None)

        while True:
            if token:
                params["token"] = token

            # Retry with exponential backoff on transient failures
            for attempt in range(4):
                try:
                    resp = requests.get(S2_BULK_URL, params=params, headers=headers, timeout=60)
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except (requests.exceptions.RequestException, ValueError) as e:
                    if attempt == 3:
                        raise
                    wait = 2 ** attempt
                    progress.update(task, description=f"Retry {attempt + 1}/3 for {venue} (waiting {wait}s)...")
                    time.sleep(wait)
            else:
                raise RuntimeError(f"Failed to fetch {venue} after retries")

            # Update total on first page
            if page == 0 and "total" in data:
                progress.update(task, total=data["total"])

            batch = data.get("data", [])
            for p in batch:
                if not p.get("abstract"):
                    continue
                pid = p["paperId"]
                authors_list = p.get("authors") or []
                papers.append({
                    "bib_key": f"s2-{pid[:12]}",
                    "title": p.get("title", ""),
                    "abstract": p["abstract"],
                    "year": p.get("year"),
                    "venue": venue,
                    "authors": ", ".join(a.get("name", "") for a in authors_list),
                    "bibtex": _make_bibtex(p, venue),
                    "source": "semantic-scholar",
                    "fetched_at": now,
                })

            progress.update(task, advance=len(batch))
            page += 1

            token = data.get("token")
            if not token:
                break
            time.sleep(1.0)  # respect rate limit

    return papers
