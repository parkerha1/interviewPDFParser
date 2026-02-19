"""Find the largest numerical value in a PDF document.

Two-stage pipeline:
  1. extract_numbers  – extracts all positive numbers from page.chars, each carrying
                        its rendered x/y coordinates for context resolution
  2. process_page     – for each number, scans upward using real coordinates to find:
                          a) the nearest column header (x-aligned)
                          b) the nearest unit keyword (when in a table context)
                        then applies a specificity hierarchy to determine the multiplier
"""

from __future__ import annotations

import logging
import re
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

logging.getLogger("pdfminer").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="pdfminer")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Token:
    """A contiguous run of non-space characters on a rendered line."""
    text: str
    x0: float
    x1: float


@dataclass
class Line:
    """One horizontal line of characters from the PDF, reconstructed from page.chars."""
    y: float
    tokens: list[Token]
    full_text: str


@dataclass
class NumberMatch:
    """A single positive number found in the document."""
    value: float
    raw_text: str
    position: int       # character offset within the flat page text (for inline scan)
    page_number: int
    x0: float           # rendered left edge on the page
    x1: float           # rendered right edge on the page
    y: float            # rendered vertical position on the page


@dataclass
class Multiplier:
    """Unit multiplier derived from surrounding context."""
    factor: float
    evidence: str       # the text that justified this multiplier


# ---------------------------------------------------------------------------
# Stage 1 – reconstruct lines from page.chars
# ---------------------------------------------------------------------------

def chars_to_lines(chars: list[dict]) -> list[Line]:
    """Group page.chars by y-coordinate into sorted Line objects.

    pdfplumber's page.chars gives one dict per character with x0, x1, top, text.
    We round 'top' to the nearest point to group characters on the same visual row.

    Tokens are split on both explicit space characters and significant x-coordinate
    gaps. Many PDFs position columns by coordinate rather than inserting space
    characters, so gap-based splitting is essential for correct column separation.
    """
    by_y: dict[int, list[dict]] = defaultdict(list)
    for c in chars:
        by_y[round(c["top"])].append(c)

    lines: list[Line] = []
    for y_key in sorted(by_y.keys()):
        row = sorted(by_y[y_key], key=lambda c: c["x0"])
        tokens: list[Token] = []
        current_text = ""
        current_x0 = 0.0
        current_x1 = 0.0

        for c in row:
            ch = c["text"]
            gap = c["x0"] - current_x1 if current_text else 0.0

            # Average character width estimate for this token
            avg_char_width = (current_x1 - current_x0) / len(current_text) if current_text else 5.0

            # Split the token if: an explicit space, or a gap wider than 1.5 average char widths
            is_gap = gap > max(avg_char_width * 1.5, 4.0)

            if ch == " " or is_gap:
                if current_text:
                    tokens.append(Token(current_text, current_x0, current_x1))
                    current_text = ""
                if ch == " ":
                    continue

            if not current_text:
                current_x0 = c["x0"]
            current_text += ch
            current_x1 = c["x1"]

        if current_text:
            tokens.append(Token(current_text, current_x0, current_x1))

        full_text = " ".join(t.text for t in tokens)
        lines.append(Line(y=float(y_key), tokens=tokens, full_text=full_text))

    return lines


# ---------------------------------------------------------------------------
# Stage 1 – number extraction from lines
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"[\d,]+\.?\d*")
_NEGATIVE_RE = re.compile(r"\(\s*([\d,]+\.?\d*)\s*\)")

# Letters adjacent to a number that signal a unit suffix — kept for context resolution
_UNIT_SUFFIXES = frozenset("MKBTmkbt")


def extract_numbers(lines: list[Line], page_number: int, page_text: str) -> list[NumberMatch]:
    """Return every positive number found in *lines*.

    Uses x/y coordinates from each Line's tokens for context resolution later.
    The flat *page_text* is used to compute the character offset (position) needed
    by detect_inline_multiplier.
    """
    negative_spans: set[tuple[int, int]] = set()
    for m in _NEGATIVE_RE.finditer(page_text):
        negative_spans.add((m.start(), m.end()))

    results: list[NumberMatch] = []

    for line in lines:
        for token in line.tokens:
            raw = token.text

            for m in _NUMBER_RE.finditer(raw):
                token_raw = m.group()
                if token_raw.replace(",", "").replace(".", "") == "":
                    continue

                # Reject numbers embedded in alphanumeric codes (e.g. "0708055F")
                # but allow unit suffixes like M/K/B/T
                before_char = raw[m.start() - 1] if m.start() > 0 else ""
                after_char = raw[m.end()] if m.end() < len(raw) else ""
                if before_char.isalpha() and before_char not in _UNIT_SUFFIXES:
                    continue
                if after_char.isalpha() and after_char not in _UNIT_SUFFIXES:
                    continue

                cleaned = token_raw.replace(",", "")
                try:
                    value = float(cleaned)
                except ValueError:
                    continue

                if value == 0:
                    continue

                # Find this token's character offset in the flat page text so
                # detect_inline_multiplier can scan forward from it
                position = page_text.find(token_raw)

                # Skip if inside an accounting-negative span in the flat text
                if any(position >= ns and position + len(token_raw) <= ne
                       for ns, ne in negative_spans):
                    continue

                results.append(NumberMatch(
                    value=value,
                    raw_text=token_raw,
                    position=position,
                    page_number=page_number,
                    x0=token.x0,
                    x1=token.x1,
                    y=line.y,
                ))

    return results


# ---------------------------------------------------------------------------
# Stage 2 – context resolution
# ---------------------------------------------------------------------------

_UNIT_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\bin\s+billions?\b", re.IGNORECASE), 1_000_000_000),
    (re.compile(r"\bin\s+millions?\b", re.IGNORECASE), 1_000_000),
    (re.compile(r"\bin\s+thousands?\b", re.IGNORECASE), 1_000),
]

_PER_RE = re.compile(r"\bper\b", re.IGNORECASE)

_INLINE_UNIT_RE = re.compile(
    r"\b(billion|million|thousand)s?\b", re.IGNORECASE
)
_INLINE_FACTORS = {"billion": 1_000_000_000, "million": 1_000_000, "thousand": 1_000}


def _parse_unit_from_text(text: str) -> float | None:
    """Return a multiplier factor if *text* contains a known unit keyword, else None."""
    for pattern, factor in _UNIT_PATTERNS:
        if pattern.search(text):
            return factor
    return None


def find_unit_context(lines: list[Line], y: float) -> Multiplier | None:
    """Scan lines above *y* (closest first) for the nearest unit keyword.

    Returns the first matching Multiplier, or None if no unit keyword is found above.
    """
    above = [ln for ln in lines if ln.y < y]
    for ln in reversed(above):
        factor = _parse_unit_from_text(ln.full_text)
        if factor is not None:
            return Multiplier(factor=factor, evidence=f"unit header: {ln.full_text!r}")
    return None


def _is_header_like(line: Line) -> bool:
    """Return True if the line looks like a column header rather than data.

    Heuristic: fewer than 30% of tokens parse as pure numbers.
    A header line mostly contains labels; a data line mostly contains numbers.
    """
    if not line.tokens:
        return False
    numeric_count = sum(
        1 for t in line.tokens
        if t.text.replace(",", "").replace(".", "").replace("-", "").isdigit()
    )
    return numeric_count / len(line.tokens) < 0.3


def find_column_header(lines: list[Line], y: float, x0: float, x1: float) -> str | None:
    """Scan lines above *y* (closest first) for the nearest column header.

    A qualifying header-like line must have at least one token whose x-range
    overlaps with [x0, x1]. When a match is found, we also collect immediately
    adjacent tokens (gap ≤ 20px) to reconstruct multi-word column phrases like
    '$ Per Barrel' where only the last token spatially overlaps the data column.
    Returns the combined phrase text, or None.
    """
    above = [ln for ln in lines if ln.y < y]
    for ln in reversed(above):
        if not _is_header_like(ln):
            continue

        # Find the index of the first overlapping token
        match_idx = None
        for i, token in enumerate(ln.tokens):
            if token.x1 >= x0 and token.x0 <= x1:
                match_idx = i
                break

        if match_idx is None:
            continue

        # Expand left: collect adjacent tokens with gap ≤ 20px
        phrase_tokens = [ln.tokens[match_idx]]
        i = match_idx - 1
        while i >= 0:
            gap = phrase_tokens[0].x0 - ln.tokens[i].x1
            if gap <= 20:
                phrase_tokens.insert(0, ln.tokens[i])
                i -= 1
            else:
                break

        # Expand right as well, for completeness
        i = match_idx + 1
        while i < len(ln.tokens):
            gap = ln.tokens[i].x0 - phrase_tokens[-1].x1
            if gap <= 20:
                phrase_tokens.append(ln.tokens[i])
                i += 1
            else:
                break

        return " ".join(t.text for t in phrase_tokens)

    return None


def detect_inline_multiplier(page_text: str, position: int) -> Multiplier | None:
    """Scan ~60 chars after *position* in the flat page text for an inline unit word.

    Handles phrases like '3.2 million dollars' or 'approximately $1.5 billion'.
    """
    window = page_text[position: position + 60]
    m = _INLINE_UNIT_RE.search(window)
    if m:
        factor = _INLINE_FACTORS[m.group(1).lower()]
        return Multiplier(factor=factor, evidence=f"inline: {window.strip()!r}")
    return None


def resolve_multiplier(
    col_header: str | None,
    unit_ctx: Multiplier | None,
    inline_mult: Multiplier | None,
) -> Multiplier:
    """Apply the specificity hierarchy to determine the multiplier for a number.

    1. Column header containing 'per'  → rate, multiplier = 1
    2. Column header containing a unit keyword → use that unit
    3. Inline cue immediately after the number
    4. Column header found (any) + unit context above → use unit context
       (col_header presence signals tabular context; prose numbers skip this step
        so they don't inherit a unit from an unrelated table elsewhere on the page)
    5. No cue → multiplier = 1
    """
    if col_header is not None:
        if _PER_RE.search(col_header):
            return Multiplier(factor=1, evidence=f"rate column: {col_header!r}")

        factor = _parse_unit_from_text(col_header)
        if factor is not None:
            return Multiplier(factor=factor, evidence=f"column header: {col_header!r}")

    if inline_mult is not None:
        return inline_mult

    if col_header is not None and unit_ctx is not None:
        return unit_ctx

    return Multiplier(factor=1, evidence="no context found")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def process_page(page: pdfplumber.page.Page) -> list[tuple[NumberMatch, Multiplier]]:
    """Extract all numbers from *page* and resolve each one's unit multiplier."""
    page_text = page.extract_text() or ""
    if not page_text.strip():
        return []

    lines = chars_to_lines(page.chars)
    numbers = extract_numbers(lines, page.page_number, page_text)

    results: list[tuple[NumberMatch, Multiplier]] = []
    for nm in numbers:
        col_header = find_column_header(lines, nm.y, nm.x0, nm.x1)
        unit_ctx   = find_unit_context(lines, nm.y)
        inline     = detect_inline_multiplier(page_text, nm.position)
        mult       = resolve_multiplier(col_header, unit_ctx, inline)
        results.append((nm, mult))

    return results


def snippet(text: str, position: int, radius: int = 60) -> str:
    """Return a short text excerpt around *position* for display."""
    start = max(0, position - radius)
    end = min(len(text), position + radius)
    return f"...{text[start:end].replace(chr(10), ' ')}..."


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
            page_text = page.extract_text() or ""
            if page_text.strip():
                page_texts[page.page_number] = page_text

            for nm, mult in process_page(page):
                if raw_max is None or nm.value > raw_max.value:
                    raw_max = nm

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
        print(f"  Multiplier:            {adj_max_mult.factor:,.0f}x ({adj_max_mult.evidence})")
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
