"""ACL Anthology data fetcher — downloads and parses the global bib file with abstracts."""

from __future__ import annotations

import gzip
import re
from datetime import datetime, timezone
from pathlib import Path

import requests
from rich.progress import Progress, SpinnerColumn, TextColumn

from ouap.config import ACL_BIB_URL, ACL_VENUES

# Regex to extract fields: handles both {value} and "value" delimiters
# For "value": captures content between double quotes (may contain braces)
_FIELD_QUOTED_RE = re.compile(r"(\w+)\s*=\s*\"((?:[^\"]|\"\")*?)\"", re.DOTALL)
# For {value}: handles up to 2 levels of nested braces
_FIELD_BRACED_RE = re.compile(
    r"(\w+)\s*=\s*\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", re.DOTALL
)

# Regex to extract the entry key: @type{key,
_KEY_RE = re.compile(r"@(\w+)\{([^,]+),")


def _split_bib_entries(text: str) -> list[str]:
    """Split a bib file into individual entries using brace-depth counting.

    This correctly handles @ signs inside field values (e.g., email addresses)
    which the old regex-based splitter could not.
    """
    entries = []
    i = 0
    n = len(text)
    while i < n:
        # Find the start of an entry: @type{
        match = _KEY_RE.search(text, i)
        if not match:
            break
        start = match.start()
        # Find the opening brace
        brace_pos = text.index("{", match.start())
        depth = 1
        j = brace_pos + 1
        while j < n and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            entries.append(text[start:j])
        i = j
    return entries


def _extract_fields(entry_text: str) -> dict[str, str]:
    """Extract all fields from a single bib entry."""
    fields = {}
    # Try quoted fields first (e.g., title = "Some Title")
    for m in _FIELD_QUOTED_RE.finditer(entry_text):
        fields[m.group(1).lower()] = m.group(2).strip()
    # Then braced fields (e.g., year = {2024}), which may override
    for m in _FIELD_BRACED_RE.finditer(entry_text):
        fields[m.group(1).lower()] = m.group(2).strip()
    return fields


def _classify_venue(booktitle: str) -> str | None:
    """Map a booktitle to a normalized venue name, or None if unrecognized.

    Findings is checked first because its booktitle contains
    'Association for Computational Linguistics' which would also match ACL.
    """
    if not booktitle:
        return None
    bt_lower = booktitle.lower()
    # Check Findings first to avoid false ACL match
    for pattern in ACL_VENUES["Findings"]["booktitle_patterns"]:
        if pattern.lower() in bt_lower:
            return "Findings"
    for venue, info in ACL_VENUES.items():
        if venue == "Findings":
            continue
        for pattern in info["booktitle_patterns"]:
            if pattern.lower() in bt_lower:
                return venue
    return None


def download_acl_bib(cache_dir: Path | None = None, force: bool = False) -> Path:
    """Download the ACL anthology+abstracts.bib.gz and return path to decompressed file.

    Skips download if a cached file exists and is less than 30 days old (unless force=True).
    """
    from platformdirs import user_cache_dir

    cache = cache_dir or Path(user_cache_dir("ouap"))
    cache.mkdir(parents=True, exist_ok=True)
    bib_path = cache / "anthology_abstracts.bib"

    # Use cached file if fresh enough
    if not force and bib_path.exists():
        age_days = (datetime.now(timezone.utc).timestamp() - bib_path.stat().st_mtime) / 86400
        if age_days < 30:
            return bib_path

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("Downloading ACL Anthology bib (this may take a minute)...", total=None)
        resp = requests.get(ACL_BIB_URL, timeout=300)
        resp.raise_for_status()

        # Decompress gzip and write to cache
        raw = gzip.decompress(resp.content)
        bib_path.write_bytes(raw)

    return bib_path


def parse_acl_bib(
    bib_path: Path,
    venues: list[str] | None = None,
    year_min: int = 2016,
    year_max: int = 2026,
) -> list[dict]:
    """Parse an ACL bib file into a list of paper dicts ready for store.upsert_papers().

    Args:
        bib_path: Path to the decompressed .bib file.
        venues: If provided, only include these ACL sub-venues.
        year_min/year_max: Year range filter.

    Returns:
        List of dicts with keys matching the papers table schema.
    """
    text = bib_path.read_text(encoding="utf-8", errors="replace")
    entries = _split_bib_entries(text)
    now = datetime.now(timezone.utc).isoformat()
    papers = []

    for entry in entries:
        key_match = _KEY_RE.match(entry)
        if not key_match:
            continue
        bib_key = key_match.group(2).strip()
        fields = _extract_fields(entry)

        # Year filter
        year_str = fields.get("year", "")
        if not year_str.isdigit():
            continue
        year = int(year_str)
        if year < year_min or year > year_max:
            continue

        # Venue classification (based on booktitle field)
        booktitle = fields.get("booktitle", "")
        venue = _classify_venue(booktitle)
        if venue is None:
            continue
        if venues and venue not in venues:
            continue

        abstract = fields.get("abstract", "")
        if not abstract:
            continue

        title = fields.get("title", "")
        authors = fields.get("author", "")

        papers.append({
            "bib_key": bib_key,
            "title": title,
            "abstract": abstract,
            "year": year,
            "venue": venue,
            "authors": authors,
            "bibtex": entry.strip(),
            "source": "acl-anthology",
            "fetched_at": now,
        })

    return papers


def fetch_acl_papers(
    venues: list[str] | None = None,
    year_min: int = 2016,
    year_max: int = 2026,
    cache_dir: Path | None = None,
) -> list[dict]:
    """Download (if needed) and parse ACL Anthology papers."""
    bib_path = download_acl_bib(cache_dir)
    return parse_acl_bib(bib_path, venues=venues, year_min=year_min, year_max=year_max)
