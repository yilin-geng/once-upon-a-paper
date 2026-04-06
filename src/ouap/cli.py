"""CLI entry point for once-upon-a-paper."""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from ouap.config import (
    ACL_VENUES,
    ALL_VENUES,
    CATEGORY_ALERTS,
    CATEGORY_COLORS,
    CATEGORY_DISPLAY_ORDER,
    CATEGORY_LABELS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL,
    DEFAULT_YEAR_MAX,
    DEFAULT_YEAR_MIN,
    S2_VENUES,
)

console = Console()


def _parse_env_value(value: str) -> str:
    """Parse a dotenv value, including standard quoted strings."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value[1:-1]
        return parsed if isinstance(parsed, str) else value
    return value


def _load_env_file(env_path: str) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ."""
    path = Path(env_path).expanduser().resolve()
    if not path.is_file():
        console.print(f"[red]Error: env file not found: {path}[/red]")
        sys.exit(1)
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = _parse_env_value(value)
        # Don't overwrite existing env vars (explicit exports take priority)
        if key not in os.environ:
            os.environ[key] = value


def _load_env_file_callback(
    ctx: click.Context, param: click.Parameter, value: str | None
) -> str | None:
    """Load env vars during option parsing so subcommands can resolve env-backed defaults."""
    del param
    if value and not ctx.resilient_parsing:
        _load_env_file(value)
    return value


def _parse_venues(venues_str: str | None) -> list[str] | None:
    """Parse comma-separated venue string. Returns None for 'all'."""
    if not venues_str or venues_str.lower() == "all":
        return None
    venues = [v.strip() for v in venues_str.split(",")]
    for v in venues:
        if v not in ALL_VENUES:
            console.print(f"[red]Unknown venue: {v}[/red]")
            console.print(f"Available: {', '.join(ALL_VENUES)}")
            sys.exit(1)
    return venues


def _parse_years(years_str: str) -> tuple[int, int]:
    """Parse 'YYYY-YYYY' year range string."""
    try:
        if "-" in years_str:
            parts = years_str.split("-", 1)
            y_min, y_max = int(parts[0]), int(parts[1])
        else:
            y_min = y_max = int(years_str)
    except ValueError:
        console.print(f"[red]Invalid year range: {years_str!r}. Expected YYYY or YYYY-YYYY.[/red]")
        sys.exit(1)
    if y_min > y_max:
        console.print(f"[red]Invalid year range: start ({y_min}) > end ({y_max}).[/red]")
        sys.exit(1)
    return y_min, y_max


@click.group()
@click.option(
    "--env-file",
    default=None,
    expose_value=False,
    is_eager=True,
    callback=_load_env_file_callback,
    help="Path to .env file with API keys (see .env.example).",
)
def main():
    """once-upon-a-paper: Find related work for your paper using LLMs."""
    pass


@main.command()
@click.option("--venues", "-v", default="all", help="Comma-separated venues or 'all'.")
@click.option("--years", "-y", default=f"{DEFAULT_YEAR_MIN}-{DEFAULT_YEAR_MAX}", help="Year range YYYY-YYYY.")
@click.option("--s2-api-key", envvar="SEMANTIC_SCHOLAR_API_KEY", default=None, help="Semantic Scholar API key.")
def fetch(venues: str, years: str, s2_api_key: str | None):
    """Download/refresh paper data from ACL Anthology and Semantic Scholar."""
    from ouap.data.store import get_connection, upsert_papers
    from ouap.data.acl import fetch_acl_papers
    from ouap.data.semantic_scholar import fetch_s2_papers

    venue_list = _parse_venues(venues)
    year_min, year_max = _parse_years(years)
    conn = get_connection()

    # Determine which ACL and S2 venues to fetch
    acl_to_fetch = []
    s2_to_fetch = []
    targets = venue_list or ALL_VENUES
    for v in targets:
        if v in ACL_VENUES:
            acl_to_fetch.append(v)
        elif v in S2_VENUES:
            s2_to_fetch.append(v)

    total_count = 0

    # Fetch ACL papers (one download covers all ACL venues)
    if acl_to_fetch:
        console.print(f"\n[bold]Fetching ACL Anthology papers for: {', '.join(acl_to_fetch)}[/bold]")
        papers = fetch_acl_papers(venues=acl_to_fetch, year_min=year_min, year_max=year_max)
        count = upsert_papers(conn, papers)
        console.print(f"  [green]✓[/green] Stored {count} ACL papers with abstracts")
        total_count += count

    # Fetch S2 papers (one API call per venue)
    for venue in s2_to_fetch:
        console.print(f"\n[bold]Fetching {venue} from Semantic Scholar...[/bold]")
        papers = fetch_s2_papers(venue=venue, year_min=year_min, year_max=year_max, api_key=s2_api_key)
        count = upsert_papers(conn, papers)
        console.print(f"  [green]✓[/green] Stored {count} {venue} papers with abstracts")
        total_count += count

    console.print(f"\n[bold green]Done![/bold green] Total: {total_count} papers stored.\n")
    conn.close()


@main.command("list-venues")
def list_venues():
    """Show available venues and cached paper counts."""
    from ouap.data.store import get_connection, get_paper_count, get_venue_year_range

    conn = get_connection()
    counts = get_paper_count(conn, venues=None, year_min=0, year_max=9999)
    year_ranges = get_venue_year_range(conn)
    conn.close()

    if not counts:
        console.print("[yellow]No data cached yet. Run 'ouap fetch' first.[/yellow]")
        return

    table = Table(title="Cached Paper Data")
    table.add_column("Venue", style="bold")
    table.add_column("Source")
    table.add_column("Papers", justify="right")
    table.add_column("Year Range", justify="center")

    for venue in ALL_VENUES:
        if venue in counts:
            source = "ACL Anthology" if venue in ACL_VENUES else "Semantic Scholar"
            yr = year_ranges.get(venue, (0, 0))
            table.add_row(venue, source, str(counts[venue]), f"{yr[0]}-{yr[1]}")

    console.print(table)


def _display_categorized_results(categorized: list[dict], paper_map: dict) -> None:
    """Display categorized results grouped by category with summary."""
    # Build per-category groups
    groups: dict[str, list[dict]] = {}
    for m in categorized:
        cat = m.get("category", "TANGENTIAL")
        groups.setdefault(cat, []).append(m)

    # Summary table
    console.print(f"\n[bold green]Found {len(categorized)} related papers![/bold green]\n")
    summary = Table(title="Categorization Summary")
    summary.add_column("Category", style="bold")
    summary.add_column("Papers", justify="right")
    summary.add_column("Avg Relevance", justify="right")
    summary.add_column("", style="bold")

    for cat in CATEGORY_DISPLAY_ORDER:
        if cat not in groups:
            continue
        items = groups[cat]
        scores = [m.get("relevance_score", 0) for m in items]
        avg = sum(scores) / len(scores)
        alert = CATEGORY_ALERTS.get(cat, "")
        color = CATEGORY_COLORS.get(cat, "")
        label = CATEGORY_LABELS.get(cat, cat)
        alert_str = f"[{color}]{alert}[/{color}]" if alert else ""
        summary.add_row(f"[{color}]{label}[/{color}]", str(len(items)), f"{avg:.1f}", alert_str)

    console.print(summary)

    # Per-category tables
    for cat in CATEGORY_DISPLAY_ORDER:
        if cat not in groups:
            continue
        items = groups[cat]
        label = CATEGORY_LABELS.get(cat, cat)
        color = CATEGORY_COLORS.get(cat, "")
        alert = CATEGORY_ALERTS.get(cat, "")
        alert_str = f"  [{color}]{alert}[/{color}]" if alert else ""

        # Sort by score descending, then year descending
        items.sort(key=lambda m: (-m.get("relevance_score", 0), -(paper_map.get(m["bib_key"], {}).get("year") or 0)))

        table = Table(title=f"{label} ({len(items)} papers){alert_str}")
        table.add_column("Score", justify="center", style="bold", width=5)
        table.add_column("Year", justify="center", width=4)
        table.add_column("Venue", width=8)
        table.add_column("Bib Key", style="bold", max_width=40)
        table.add_column("Reason", max_width=55)

        for m in items:
            paper = paper_map.get(m["bib_key"], {})
            table.add_row(
                str(m.get("relevance_score", "")),
                str(paper.get("year", "")),
                paper.get("venue", ""),
                m["bib_key"],
                m.get("reason", ""),
            )

        console.print(f"\n")
        console.print(table)


@main.command()
@click.option("--abstract", "-a", required=True, help="Path to abstract text file, or '-' for stdin.")
@click.option("--venues", "-v", default="all", help="Comma-separated venues or 'all'.")
@click.option("--years", "-y", default=f"{DEFAULT_YEAR_MIN}-{DEFAULT_YEAR_MAX}", help="Year range YYYY-YYYY.")
@click.option("--model", "-m", default=DEFAULT_MODEL, help="Model name.")
@click.option("--api-key", envvar="LLM_API_KEY", default=None, help="API key (or set OPENAI_API_KEY / ANTHROPIC_API_KEY).")
@click.option("--output", "-o", default="./results", help="Output directory.")
@click.option("--batch-size", "-b", default=DEFAULT_BATCH_SIZE, help="Papers per LLM batch.")
@click.option("--dry-run", is_flag=True, help="Show cost estimate only, do not scan.")
@click.option("--yes", is_flag=True, help="Skip cost confirmation prompt.")
@click.option("--skip-categorize", is_flag=True, help="Skip categorization, output flat list only.")
@click.option("--no-report", is_flag=True, help="Skip report generation (Phase 3).")
def scan(
    abstract: str,
    venues: str,
    years: str,
    model: str,
    api_key: str | None,
    output: str,
    batch_size: int,
    dry_run: bool,
    yes: bool,
    skip_categorize: bool,
    no_report: bool,
):
    """Scan papers to find related work for your abstract."""
    from ouap.cost import estimate_all_models_cost
    from ouap.data.store import get_connection, get_paper_count, get_papers_by_keys
    from ouap.output import write_bib_file, write_json_report
    from ouap.scanner.categorizer import categorize
    from ouap.scanner.engine import scan as run_scan
    from ouap.scanner.providers import get_provider

    # Read abstract
    if abstract == "-":
        user_abstract = sys.stdin.read().strip()
    else:
        user_abstract = Path(abstract).read_text(encoding="utf-8").strip()

    if not user_abstract:
        console.print("[red]Error: empty abstract.[/red]")
        sys.exit(1)

    console.print(f"\n[bold]Proposed Abstract[/bold] ({len(user_abstract.split())} words):")
    preview = user_abstract[:300] + ("..." if len(user_abstract) > 300 else "")
    console.print(f"  [dim]{preview}[/dim]\n")

    venue_list = _parse_venues(venues)
    year_min, year_max = _parse_years(years)
    conn = get_connection()

    # Check data availability
    counts = get_paper_count(conn, venue_list, year_min, year_max)
    if not counts:
        console.print("[red]No papers found in cache. Run 'ouap fetch' first.[/red]")
        conn.close()
        sys.exit(1)

    # Show scope
    scope_table = Table(title="Scan Scope")
    scope_table.add_column("Venue", style="bold")
    scope_table.add_column("Papers", justify="right")
    total_papers = 0
    for v, c in sorted(counts.items()):
        scope_table.add_row(v, str(c))
        total_papers += c
    num_batches = (total_papers + batch_size - 1) // batch_size
    scope_table.add_section()
    scope_table.add_row("[bold]Total[/bold]", f"[bold]{total_papers}[/bold]")
    console.print(scope_table)
    console.print(f"  Batches: {num_batches} (batch size: {batch_size})\n")

    # Show cost estimate for all models
    all_costs = estimate_all_models_cost(conn, user_abstract, venue_list, year_min, year_max, batch_size)
    cost_table = Table(title="Cost Estimates (scan + categorization)")
    cost_table.add_column("Model")
    cost_table.add_column("Provider")
    cost_table.add_column("Est. Input Tokens", justify="right")
    cost_table.add_column("Est. Cost (USD)", justify="right")
    for c in all_costs:
        marker = " [bold green]◄[/bold green]" if c["model"] == model else ""
        cost_table.add_row(
            c["model"] + marker,
            c["provider"],
            f"~{c['input_tokens']:,}",
            f"${c['cost_usd']:.2f}",
        )
    console.print(cost_table)

    if dry_run:
        conn.close()
        return

    # Resolve API key
    resolved_key = api_key
    if not resolved_key:
        if model.startswith("gemini-"):
            resolved_key = os.environ.get("GOOGLE_API_KEY")
        elif model.startswith("gpt-"):
            resolved_key = os.environ.get("OPENAI_API_KEY")
        elif model.startswith("claude-"):
            resolved_key = os.environ.get("ANTHROPIC_API_KEY")

    if not resolved_key and model != "local" and not model.startswith("meta-"):
        if model.startswith("gemini-"):
            env_var = "GOOGLE_API_KEY"
        elif model.startswith("gpt-"):
            env_var = "OPENAI_API_KEY"
        else:
            env_var = "ANTHROPIC_API_KEY"
        console.print(f"[red]Error: No API key found. Set {env_var} or pass --api-key.[/red]")
        conn.close()
        sys.exit(1)

    # Confirm
    if not yes:
        selected = next((c for c in all_costs if c["model"] == model), None)
        cost_str = f"${selected['cost_usd']:.2f}" if selected else "unknown"
        if not click.confirm(f"\nProceed with {model} (est. {cost_str})?", default=True):
            conn.close()
            return

    # Phase 1: Discovery scan
    console.print(f"\n[bold]Phase 1: Scanning with {model}...[/bold]\n")
    provider = get_provider(model, api_key=resolved_key)
    matches = run_scan(conn, user_abstract, venue_list, year_min, year_max, provider, batch_size)

    if not matches:
        console.print("[yellow]No related papers found.[/yellow]")
        conn.close()
        return

    console.print(f"\n[bold green]Found {len(matches)} related papers.[/bold green]")

    # Phase 2: Categorization
    if skip_categorize:
        # Flat display (old behavior)
        console.print()
        results_table = Table(title="Related Papers")
        results_table.add_column("Bib Key", style="bold", max_width=40)
        results_table.add_column("Reason", max_width=60)
        for m in matches:
            results_table.add_row(m["bib_key"], m["reason"])
        console.print(results_table)
        final_matches = matches
    else:
        console.print(f"\n[bold]Phase 2: Categorizing {len(matches)} papers...[/bold]\n")
        categorized = categorize(conn, user_abstract, matches, provider)

        # Build paper_map for display
        bib_keys = [m["bib_key"] for m in categorized]
        papers_for_display = get_papers_by_keys(conn, bib_keys)
        paper_map = {p["bib_key"]: p for p in papers_for_display}

        _display_categorized_results(categorized, paper_map)
        final_matches = categorized

    # Phase 3: Report generation
    report_md = None
    if not skip_categorize and not no_report:
        console.print(f"\n[bold]Phase 3: Generating report...[/bold]\n")
        from ouap.scanner.reporter import generate_report

        report_md = generate_report(
            conn, user_abstract, final_matches, provider,
            venues_scanned=venue_list, year_min=year_min, year_max=year_max,
        )

    # Write output files (timestamped subdirectory so runs coexist)
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    bib_path = out_dir / "related.bib"
    json_path = out_dir / "related.json"

    bib_count = write_bib_file(conn, final_matches, bib_path)
    json_count = write_json_report(conn, final_matches, json_path)

    console.print(f"\n[bold]Output written:[/bold]")
    console.print(f"  {bib_path}  ({bib_count} entries)")
    console.print(f"  {json_path} ({json_count} entries)")

    if report_md is not None:
        from ouap.output import write_report_file

        report_path = out_dir / "report.md"
        write_report_file(report_md, report_path)
        console.print(f"  {report_path}")

    console.print()

    conn.close()


@main.command()
@click.option("--abstract", "-a", required=True, help="Path to abstract text file.")
@click.option("--venues", "-v", default="all", help="Comma-separated venues or 'all'.")
@click.option("--years", "-y", default=f"{DEFAULT_YEAR_MIN}-{DEFAULT_YEAR_MAX}", help="Year range YYYY-YYYY.")
@click.option("--batch-size", "-b", default=DEFAULT_BATCH_SIZE, help="Papers per LLM batch.")
def cost(abstract: str, venues: str, years: str, batch_size: int):
    """Show cost estimates across all models without scanning."""
    from ouap.cost import estimate_all_models_cost
    from ouap.data.store import get_connection

    user_abstract = Path(abstract).read_text(encoding="utf-8").strip()
    venue_list = _parse_venues(venues)
    year_min, year_max = _parse_years(years)
    conn = get_connection()

    all_costs = estimate_all_models_cost(conn, user_abstract, venue_list, year_min, year_max, batch_size)

    if not all_costs:
        console.print("[yellow]No papers found. Run 'ouap fetch' first.[/yellow]")
        conn.close()
        return

    table = Table(title="Cost Estimates")
    table.add_column("Model")
    table.add_column("Provider")
    table.add_column("Input Tokens", justify="right")
    table.add_column("Output Tokens", justify="right")
    table.add_column("Est. Cost (USD)", justify="right")
    for c in all_costs:
        table.add_row(
            c["model"],
            c["provider"],
            f"~{c['input_tokens']:,}",
            f"~{c['output_tokens']:,}",
            f"${c['cost_usd']:.2f}",
        )
    console.print(table)
    conn.close()


if __name__ == "__main__":
    main()
