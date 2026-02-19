"""Find the largest numerical value in a PDF document.

Two-stage pipeline:
  1. extract_numbers  – regex-based extraction of all positive numbers per page
  2. context_multiplier – (stub) resolve natural-language unit hints near each number
"""

from __future__ import annotations

import logging
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

logging.getLogger("pdfminer").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="pdfminer")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class NumberMatch:
    """A single number found in the document."""
    value: float
    raw_text: str
    position: int          # character offset within the page text
    page_number: int


@dataclass
class Multiplier:
    """Unit multiplier derived from surrounding context."""
    factor: float
    evidence: str          # the text that justified this multiplier


# ---------------------------------------------------------------------------
# Stage 1 – number extraction
# ---------------------------------------------------------------------------

# Matches integers and decimals with optional comma grouping:
#   28,239.2   3.15   1,000   42
_NUMBER_RE = re.compile(r"[\d,]+\.?\d*")

# Accounting-style negatives: (364.7)
_NEGATIVE_RE = re.compile(r"\(\s*([\d,]+\.?\d*)\s*\)")

# Letters adjacent to a number that signal a unit suffix — keep these for context_multiplier
_UNIT_SUFFIXES = frozenset("MKBTmkbt")


def extract_numbers(text: str, page_number: int) -> list[NumberMatch]:
    """Return every positive number found in *text*."""
    negative_spans: set[tuple[int, int]] = set()
    for m in _NEGATIVE_RE.finditer(text):
        negative_spans.add((m.start(), m.end()))

    results: list[NumberMatch] = []
    for m in _NUMBER_RE.finditer(text):
        if _inside_negative(m.start(), m.end(), negative_spans):
            continue

        raw = m.group()
        if raw.replace(",", "").replace(".", "") == "":
            continue

        # Skip digit runs adjacent to non-unit letters — those are alphanumeric codes
        # (e.g. "0708055F"). Unit suffixes M/K/B/T are kept for context_multiplier to handle.
        end_pos = m.end()
        if end_pos < len(text) and text[end_pos].isalpha() and text[end_pos] not in _UNIT_SUFFIXES:
            continue
        start_pos = m.start()
        if start_pos > 0 and text[start_pos - 1].isalpha() and text[start_pos - 1] not in _UNIT_SUFFIXES:
            continue

        cleaned = raw.replace(",", "")
        try:
            value = float(cleaned)
        except ValueError:
            continue

        if value == 0:
            continue

        results.append(NumberMatch(
            value=value,
            raw_text=raw,
            position=m.start(),
            page_number=page_number,
        ))

    return results


def _inside_negative(start: int, end: int, neg_spans: set[tuple[int, int]]) -> bool:
    """Check whether a number match falls inside an accounting-negative span."""
    for ns, ne in neg_spans:
        if start >= ns and end <= ne:
            return True
    return False


# ---------------------------------------------------------------------------
# Stage 2 – context-aware multiplier (stub)
# ---------------------------------------------------------------------------

def context_multiplier(text: str, position: int) -> Multiplier:
    """Determine the unit multiplier for the number at *position* in *text*.

    TODO: inspect surrounding text for cues like "in millions",
    "in thousands", "$ in billions", etc.  This should handle:
      - page-level headers  (e.g. "(Dollars in Millions)" at top of page)
      - table-level headers (e.g. column header "($ in Thousands)")
      - inline cues         (e.g. "approximately $3.2 million")
    For now returns 1× unconditionally.
    """
    return Multiplier(factor=1, evidence="no context applied")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def snippet(text: str, position: int, radius: int = 60) -> str:
    """Return a short text excerpt around *position* for display."""
    start = max(0, position - radius)
    end = min(len(text), position + radius)
    fragment = text[start:end].replace("\n", " ")
    return f"...{fragment}..."


def find_largest(pdf_path: str) -> None:
    path = Path(pdf_path)
    if not path.exists():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    raw_max: NumberMatch | None = None
    adj_max_value: float = 0.0
    adj_max_match: NumberMatch | None = None
    adj_max_mult: Multiplier | None = None

    page_texts: dict[int, str] = {}

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_num = page.page_number        # 1-indexed
            text = page.extract_text() or ""
            if not text.strip():
                continue

            page_texts[page_num] = text
            numbers = extract_numbers(text, page_num)

            for nm in numbers:
                if raw_max is None or nm.value > raw_max.value:
                    raw_max = nm

                mult = context_multiplier(text, nm.position)
                adjusted = nm.value * mult.factor

                if adjusted > adj_max_value:
                    adj_max_value = adjusted
                    adj_max_match = nm
                    adj_max_mult = mult

    print("=" * 64)
    print("RESULTS")
    print("=" * 64)

    if raw_max is None:
        print("No numbers found in the document.")
        return

    print(f"\nLargest raw number:      {raw_max.value:,.2f}")
    print(f"  Raw text:              {raw_max.raw_text!r}")
    print(f"  Page:                  {raw_max.page_number}")
    print(f"  Context:               {snippet(page_texts[raw_max.page_number], raw_max.position)}")

    if adj_max_match and adj_max_mult:
        print(f"\nLargest adjusted number: {adj_max_value:,.2f}")
        print(f"  Raw text:              {adj_max_match.raw_text!r}")
        print(f"  Multiplier:            {adj_max_mult.factor}x ({adj_max_mult.evidence})")
        print(f"  Page:                  {adj_max_match.page_number}")
        print(f"  Context:               {snippet(page_texts[adj_max_match.page_number], adj_max_match.position)}")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path-to-pdf>", file=sys.stderr)
        sys.exit(1)

    find_largest(sys.argv[1])
