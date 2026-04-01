"""Token counting and cost estimation."""

from __future__ import annotations

import sqlite3

from ouap.config import DEFAULT_BATCH_SIZE, DEFAULT_CATEGORIZE_BATCH_SIZE, MODEL_PRICING
from ouap.data.store import query_papers
from ouap.scanner.categorizer import SYSTEM_PROMPT_CATEGORIZE
from ouap.scanner.prompt import SYSTEM_PROMPT
from ouap.scanner.providers import LLMProvider


def estimate_scan_cost(
    conn: sqlite3.Connection,
    user_abstract: str,
    venues: list[str] | None,
    year_min: int,
    year_max: int,
    provider: LLMProvider,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    """Compute estimated cost for a scan.

    Returns dict with: num_papers, num_batches, input_tokens, output_tokens, cost_usd.
    """
    papers = query_papers(conn, venues, year_min, year_max)
    num_papers = len(papers)
    if num_papers == 0:
        return {
            "num_papers": 0,
            "num_batches": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
        }

    num_batches = (num_papers + batch_size - 1) // batch_size

    # Count tokens via sampling for speed
    system_tokens = provider.count_tokens(SYSTEM_PROMPT)
    abstract_tokens = provider.count_tokens(user_abstract)

    # Sample up to 100 papers for average token count
    sample = papers[:100]
    sample_tokens = 0
    for p in sample:
        text = f"bib_key: {p['bib_key']}\nTitle: {p['title']}\nAbstract: {p.get('abstract', '')}\n"
        sample_tokens += provider.count_tokens(text)
    avg_paper_tokens = sample_tokens / len(sample)
    total_paper_tokens = int(avg_paper_tokens * num_papers)

    # Per-batch overhead: system + abstract + formatting
    per_batch_overhead = system_tokens + abstract_tokens + 50
    total_input = per_batch_overhead * num_batches + total_paper_tokens

    # Output: estimate ~20% match rate, ~30 tokens per match
    estimated_matches = int(num_papers * 0.2)
    total_output = estimated_matches * 30 + num_batches * 5

    cost = (
        total_input * provider.cost_per_input_token
        + total_output * provider.cost_per_output_token
    )

    return {
        "num_papers": num_papers,
        "num_batches": num_batches,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "cost_usd": cost,
    }


def estimate_all_models_cost(
    conn: sqlite3.Connection,
    user_abstract: str,
    venues: list[str] | None,
    year_min: int,
    year_max: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[dict]:
    """Estimate cost across all configured models. Returns list of {model, cost_usd, input_tokens}."""
    papers = query_papers(conn, venues, year_min, year_max)
    num_papers = len(papers)
    if num_papers == 0:
        return []

    num_batches = (num_papers + batch_size - 1) // batch_size

    # Approximate token counts (use ~4 chars per token as universal estimate)
    system_tokens = len(SYSTEM_PROMPT) // 4
    abstract_tokens = len(user_abstract) // 4

    sample = papers[:100]
    sample_chars = sum(
        len(f"bib_key: {p['bib_key']}\nTitle: {p['title']}\nAbstract: {p.get('abstract', '')}\n")
        for p in sample
    )
    avg_paper_tokens = (sample_chars / len(sample)) / 4
    total_paper_tokens = int(avg_paper_tokens * num_papers)

    per_batch_overhead = system_tokens + abstract_tokens + 50
    scan_input = per_batch_overhead * num_batches + total_paper_tokens
    estimated_matches = int(num_papers * 0.2)
    scan_output = estimated_matches * 30 + num_batches * 5

    # Categorization pass: only processes matched papers
    cat_system_tokens = len(SYSTEM_PROMPT_CATEGORIZE) // 4
    cat_batch_size = DEFAULT_CATEGORIZE_BATCH_SIZE
    cat_num_batches = (estimated_matches + cat_batch_size - 1) // cat_batch_size
    # Each matched paper: ~150 tokens (bib_key + title + truncated abstract + scan reason)
    cat_per_paper_tokens = 150
    cat_per_batch_overhead = cat_system_tokens + abstract_tokens + 50
    cat_input = cat_per_batch_overhead * cat_num_batches + estimated_matches * cat_per_paper_tokens
    # Output: ~20 tokens per paper (bib_key | CATEGORY | score | reason)
    cat_output = estimated_matches * 20

    # Report generation: 5 LLM calls, ~3000 input tokens + ~400 output tokens each
    report_input = 5 * 3000
    report_output = 5 * 400

    total_input = scan_input + cat_input + report_input
    total_output = scan_output + cat_output + report_output

    results = []
    for model, pricing in MODEL_PRICING.items():
        cost = total_input * pricing["input"] + total_output * pricing["output"]
        results.append({
            "model": model,
            "provider": pricing.get("provider", "unknown"),
            "input_tokens": total_input,
            "output_tokens": total_output,
            "cost_usd": cost,
        })

    results.sort(key=lambda x: x["cost_usd"])
    return results
