from __future__ import annotations

import heapq
import logging
import sys
import warnings
from pathlib import Path

import pdfplumber

from pdf_context import process_page
from pdf_models import Multiplier, NumberMatch

logging.getLogger("pdfminer").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="pdfminer")


def snippet(text: str, position: int, radius: int = 60) -> str:
    """Return a short text excerpt around *position* for display."""
    start = max(0, position - radius)
    end = min(len(text), position + radius)
    return f"...{text[start:end].replace(chr(10), ' ')}..."


def _print_entry(
    rank: int,
    label: str,
    nm: NumberMatch,
    mult: Multiplier,
    adjusted: float,
    page_texts: dict[int, str],
    verbose: bool,
) -> None:
    prefix = f"  #{rank}" if rank else " "
    print(f"{prefix} {label}: {adjusted:>24,.2f}")
    print(f"      Raw text:    {nm.raw_text!r}  (page {nm.page_number})")
    if mult.factor != 1:
        print(f"      Multiplier:  {mult.factor:,.0f}x  ({mult.evidence})")
    if verbose:
        if mult.factor == 1:
            print(f"      Multiplier:  1x  ({mult.evidence})")
        if nm.page_number in page_texts:
            print(f"      Context:     {snippet(page_texts[nm.page_number], nm.position)}")


def find_largest(
    pdf_path: str,
    top_n: int = 1,
    include_negatives: bool = False,
    verbose: bool = False,
) -> None:
    path = Path(pdf_path)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    raw_heap: list[tuple[float, int, NumberMatch]] = []
    adj_heap: list[tuple[float, int, NumberMatch, Multiplier]] = []
    page_texts: dict[int, str] = {}
    counter = 0

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                page_texts[page.page_number] = page_text

            for nm, mult in process_page(page, include_negatives, page_text):
                counter += 1
                adjusted = nm.value * mult.factor
                raw_key = abs(nm.value) if include_negatives else nm.value
                adj_key = abs(adjusted) if include_negatives else adjusted

                if len(raw_heap) < top_n:
                    heapq.heappush(raw_heap, (raw_key, counter, nm))
                elif raw_key > raw_heap[0][0]:
                    heapq.heapreplace(raw_heap, (raw_key, counter, nm))

                if len(adj_heap) < top_n:
                    heapq.heappush(adj_heap, (adj_key, counter, nm, mult))
                elif adj_key > adj_heap[0][0]:
                    heapq.heapreplace(adj_heap, (adj_key, counter, nm, mult))

    print("=" * 64)
    print("RESULTS")
    print("=" * 64)

    if not raw_heap:
        print("No numbers found in the document.")
        return

    raw_sorted = sorted(raw_heap, key=lambda t: t[0], reverse=True)
    adj_sorted = sorted(adj_heap, key=lambda t: t[0], reverse=True)

    show_rank = top_n > 1
    neg_note = " (including negatives by magnitude)" if include_negatives else ""

    print(f"\nLargest raw number{'s' if top_n > 1 else ''}{neg_note}:\n")
    for i, (_, _, nm) in enumerate(raw_sorted, 1):
        no_mult = Multiplier(factor=1, evidence="raw")
        _print_entry(i if show_rank else 0, "Raw", nm, no_mult, nm.value, page_texts, verbose)

    print(f"\nLargest adjusted number{'s' if top_n > 1 else ''}{neg_note}:\n")
    for i, (adj_val, _, nm, mult) in enumerate(adj_sorted, 1):
        _print_entry(i if show_rank else 0, "Adjusted", nm, mult, adj_val, page_texts, verbose)

    print()
