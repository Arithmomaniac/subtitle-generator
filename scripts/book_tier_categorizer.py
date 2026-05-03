"""Development CLI for evidence-aware source-book tier categorization."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
import sys
from typing import Any, Literal

from subtitle_generator.copilot_web_search import CopilotMCPWebSearch
from subtitle_generator.market_tiers import source_label_tier_definitions
from subtitle_generator.parameter_state import DEFAULT_RATER_MODEL


HOSTED_WEB_SEARCH_TOOL = {"type": "web_search", "search_context_size": "low"}


def categorize_book_with_hosted_responses_search(
    *,
    title: str,
    subtitle: str,
    book_id: str = "1",
    model: str = DEFAULT_RATER_MODEL,
) -> dict[str, Any]:
    """Categorize one book using Responses hosted web_search and structured output."""

    return asyncio.run(_categorize_book_with_hosted_responses_search_async(
        {"id": book_id, "title": title, "subtitle": subtitle},
        model=model,
    ))


def categorize_books_with_hosted_responses_search(
    books: list[dict[str, str]],
    *,
    model: str = DEFAULT_RATER_MODEL,
    max_concurrency: int = 5,
) -> list[dict[str, Any]]:
    """Bulk categorize books as parallel single-book Responses calls."""

    if not books:
        return []
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")
    normalized_books = [
        _normalize_book_input(book, index)
        for index, book in enumerate(books, start=1)
    ]
    results = asyncio.run(_categorize_books_with_hosted_responses_search_async(
        normalized_books,
        model=model,
        max_concurrency=max_concurrency,
    ))
    return _validate_bulk_predictions(normalized_books, results)


def categorize_books_with_prefetched_web(
    books: list[dict[str, str]],
    *,
    model: str = DEFAULT_RATER_MODEL,
    searches_per_book: int = 1,
    batch_size: int = 10,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Fallback path: prefetch web evidence, then batch LLM classification."""

    if not books:
        return []
    if searches_per_book not in {1, 2}:
        raise ValueError("searches_per_book must be 1 or 2")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    normalized_books = [
        _normalize_book_input(book, index)
        for index, book in enumerate(books, start=1)
    ]
    search_client = CopilotMCPWebSearch()
    results: list[dict[str, Any]] = []
    for start in range(0, len(normalized_books), batch_size):
        chunk = normalized_books[start : start + batch_size]
        evidence_by_id: dict[str, str] = {}
        for book in chunk:
            evidence_by_id[book["id"]] = _prefetch_book_evidence(
                search_client=search_client,
                title=book["title"],
                subtitle=book["subtitle"],
                searches_per_book=searches_per_book,
                verbose=verbose,
            )
        results.extend(_classify_books_from_evidence(
            books=chunk,
            evidence_by_id=evidence_by_id,
            model=model,
        ))
    return results


async def _categorize_books_with_hosted_responses_search_async(
    books: list[dict[str, str]],
    *,
    model: str,
    max_concurrency: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max_concurrency)

    async def classify_with_limit(book: dict[str, str]) -> dict[str, Any]:
        async with semaphore:
            return await _categorize_book_with_hosted_responses_search_async(
                book,
                model=model,
            )

    return await asyncio.gather(*(classify_with_limit(book) for book in books))


async def _categorize_book_with_hosted_responses_search_async(
    book: dict[str, str],
    *,
    model: str,
) -> dict[str, Any]:
    try:
        import litellm
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError("LiteLLM and Pydantic are required for classification.") from exc

    class BookTier(BaseModel):
        id: str
        title: str
        tier: Literal["pop", "mainstream", "niche"]
        confidence: float = Field(ge=0.0, le=1.0)
        rationale: str
        evidence_status: Literal["matched", "weak_match", "no_match"]
        evidence_note: str

    prompt = f"""Use hosted web search before answering. Classify this real book into exactly one market tier: pop, mainstream, or niche.

{source_label_tier_definitions()}

Search/evidence rules:
- Do at least one web search for this exact title/subtitle.
- Do at most two web_search calls total for this book. Do not keep searching.
- evidence_status must be:
  - matched: search clearly found this exact book.
  - weak_match: search found adjacent or partial evidence, such as author/publisher/catalog/search-summary context.
  - no_match: no reliable matching evidence was found after bounded search.
- If evidence_status is no_match, classify as niche unless title/subtitle strongly indicates broad trade appeal; lower confidence.
- evidence_note must be one short sentence summarizing the evidence status; do not include URLs.
- Rationale must be 1-2 concise sentences.

Book:
ID: {book["id"]}
Title: {book["title"]}
Subtitle: {book["subtitle"]}
"""

    response = await litellm.aresponses(
        model=model,
        input=prompt,
        tools=[HOSTED_WEB_SEARCH_TOOL],
        tool_choice="required",
        text_format=BookTier,
        max_output_tokens=1200,
        timeout=120.0,
    )
    search_queries = _extract_responses_search_queries(response.output)
    for item in response.output:
        if hasattr(item, "content") and item.content:
            for content in item.content:
                if hasattr(content, "text"):
                    result = BookTier.model_validate_json(content.text).model_dump()
                    result["search_queries"] = search_queries
                    return result
    raise RuntimeError(f"No structured text returned for book id {book['id']}")


def _classify_books_from_evidence(
    *,
    books: list[dict[str, str]],
    evidence_by_id: dict[str, str],
    model: str,
) -> list[dict[str, Any]]:
    try:
        import litellm
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError("LiteLLM and Pydantic are required for classification.") from exc

    class BookTierPrediction(BaseModel):
        id: str
        title: str
        tier: Literal["pop", "mainstream", "niche"]
        confidence: float = Field(ge=0.0, le=1.0)
        rationale: str
        evidence_status: Literal["matched", "weak_match", "no_match"]
        evidence_note: str

    class BookTierBatch(BaseModel):
        predictions: list[BookTierPrediction]

    book_blocks = []
    for book in books:
        evidence = evidence_by_id[book["id"]][:6000]
        book_blocks.append(
            f"""ID: {book["id"]}
Title: {book["title"]}
Subtitle: {book["subtitle"]}
Web evidence:
{evidence}"""
        )

    prompt = f"""Classify each real book into exactly one market tier: pop, mainstream, or niche.

{source_label_tier_definitions()}

Return exactly one prediction for each input ID. Preserve each input ID exactly.
Each rationale must be 1-2 concise sentences explaining why the selected tier fits.
Set evidence_status to matched, weak_match, or no_match, and include a one-sentence evidence_note without URLs.

Books:
{"\n\n---\n\n".join(book_blocks)}
"""

    short_model = model.rsplit("/", 1)[-1]
    if short_model in {"gpt-5.4-mini", "gpt-5.4", "gpt-5.4-nano"}:
        response = asyncio.run(litellm.aresponses(
            model=model,
            input=prompt,
            text_format=BookTierBatch,
            max_output_tokens=max(1200, 450 * len(books)),
            timeout=120.0,
        ))
        for item in response.output:
            if hasattr(item, "content") and item.content:
                for content in item.content:
                    if hasattr(content, "text"):
                        batch = BookTierBatch.model_validate_json(content.text)
                        return _validate_bulk_predictions(books, batch.model_dump()["predictions"])
        raise RuntimeError("No structured text returned from Responses API")

    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=BookTierBatch,
        temperature=0,
    )
    batch = BookTierBatch.model_validate_json(response.choices[0].message.content)
    return _validate_bulk_predictions(books, batch.model_dump()["predictions"])


def _prefetch_book_evidence(
    *,
    search_client: CopilotMCPWebSearch,
    title: str,
    subtitle: str,
    searches_per_book: int,
    verbose: bool,
) -> str:
    queries = [
        f"{title} {subtitle} book publisher author reviews audience market readership academic trade nonfiction",
        f"{title} {subtitle} publisher page reviews academic monograph trade general readers",
    ][:searches_per_book]
    evidence_parts: list[str] = []
    for query in queries:
        if verbose:
            print(f"[search] {query}", file=sys.stderr)
        evidence_parts.append(search_client.search(query).text)
    return "\n\n".join(evidence_parts)


def _extract_responses_search_queries(output: list[Any]) -> list[list[str]]:
    search_queries: list[list[str]] = []
    for item in output:
        if getattr(item, "type", None) != "web_search_call":
            continue
        action = getattr(item, "action", None)
        queries = getattr(action, "queries", None)
        if isinstance(queries, list) and queries:
            search_queries.append([str(query) for query in queries])
            continue
        query = getattr(action, "query", None)
        if query:
            search_queries.append([str(query)])
    return search_queries


def _normalize_book_input(book: dict[str, str], index: int) -> dict[str, str]:
    title = str(book.get("title", "")).strip()
    if not title:
        raise ValueError(f"book #{index} is missing title")
    return {
        "id": str(book.get("id") or index),
        "title": title,
        "subtitle": str(book.get("subtitle") or "").strip(),
    }


def _validate_bulk_predictions(
    books: list[dict[str, str]],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expected = Counter(book["id"] for book in books)
    actual = Counter(str(prediction.get("id", "")) for prediction in predictions)
    if actual != expected:
        missing = sorted((expected - actual).elements())
        extra = sorted((actual - expected).elements())
        raise RuntimeError(
            "Bulk classification returned mismatched IDs: "
            f"missing={missing}, extra={extra}"
        )
    order = {book["id"]: index for index, book in enumerate(books)}
    return sorted(predictions, key=lambda prediction: order[str(prediction["id"])])


def _load_books(path: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as handle:
        if path.lower().endswith(".jsonl"):
            return [
                json.loads(line)
                for line in handle
                if line.strip()
            ]
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("bulk input JSON must be an array of book objects")
    return data


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    cat_parser = subparsers.add_parser("categorize", help="Categorize one book")
    cat_parser.add_argument("--title", required=True)
    cat_parser.add_argument("--subtitle", required=True)
    cat_parser.add_argument("--model", default=DEFAULT_RATER_MODEL)

    bulk_parser = subparsers.add_parser(
        "categorize-bulk",
        help="Categorize many books with hosted Responses web_search and bounded parallelism",
    )
    bulk_parser.add_argument(
        "--input",
        required=True,
        help="JSON array or JSONL of objects with title, optional subtitle, optional id.",
    )
    bulk_parser.add_argument("--model", default=DEFAULT_RATER_MODEL)
    bulk_parser.add_argument(
        "--strategy",
        choices=["hosted", "prefetch"],
        default="hosted",
        help="hosted runs one searched Responses call per book; prefetch batches after deterministic MCP searches.",
    )
    bulk_parser.add_argument("--max-concurrency", type=int, default=5)
    bulk_parser.add_argument("--batch-size", type=int, default=10)
    bulk_parser.add_argument("--searches-per-book", type=int, choices=[1, 2], default=1)
    bulk_parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "categorize":
        result = categorize_book_with_hosted_responses_search(
            title=args.title,
            subtitle=args.subtitle,
            model=args.model,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.command == "categorize-bulk":
        books = _load_books(args.input)
        if args.strategy == "prefetch":
            result = categorize_books_with_prefetched_web(
                books,
                model=args.model,
                searches_per_book=args.searches_per_book,
                batch_size=args.batch_size,
                verbose=args.verbose,
            )
        else:
            result = categorize_books_with_hosted_responses_search(
                books,
                model=args.model,
                max_concurrency=args.max_concurrency,
            )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
