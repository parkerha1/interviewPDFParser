"""Find the largest numerical value in a PDF document.

Two-stage pipeline:
  1. extract_numbers      – extracts all positive numbers from page.chars, each
                            carrying its rendered x/y coordinates for context resolution
  2. process_page         – for each number:
       a) find_column_header       – scan upward for the nearest x-aligned header
       b) find_scoped_unit_context – match against spatially-scoped unit sections
          (unit headers like "(Dollars in Millions)" are bounded by horizontal rules
          so they only apply to numbers within their table section, not the whole page)
       c) detect_inline_multiplier – check for inline unit words after the number
       d) resolve_multiplier       – apply specificity hierarchy
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


# ---------------------------------------------------------------------------
# Stage 2 – spatial section scoping via horizontal rules
# ---------------------------------------------------------------------------

_HEADER_RULE_GAP = 30  # pt — consecutive rules within this gap are header borders


@dataclass
class UnitSection:
    """A unit header and the y-range it owns on a page."""
    y_header: float
    factor: float
    evidence: str
    data_start: float
    section_end: float


def get_horizontal_rules(page: pdfplumber.page.Page, min_width_frac: float = 0.3) -> list[float]:
    """Return sorted, deduplicated y-positions of horizontal rules on *page*.

    Only includes rules wider than *min_width_frac* of the page width.
    Rules within 3pt of each other are merged.
    """
    min_width = page.width * min_width_frac
    rule_ys: set[float] = set()
    for edge in (page.edges or []):
        if abs(edge.get("top", 0) - edge.get("bottom", 0)) < 2:
            if edge.get("x1", 0) - edge.get("x0", 0) >= min_width:
                rule_ys.add(round(edge["top"], 1))

    deduped: list[float] = []
    for y in sorted(rule_ys):
        if not deduped or y - deduped[-1] > 3:
            deduped.append(y)
    return deduped


def _find_unit_scope(rules: list[float], y_unit: float, page_height: float) -> tuple[float, float]:
    """Determine the data y-range that a unit header at *y_unit* owns.

    The scope extends downward from the last "header rule" (consecutive rules
    within _HEADER_RULE_GAP of each other immediately below the header) to the
    next rule after that (or the page bottom).
    """
    rules_below = [r for r in rules if r > y_unit]
    if not rules_below:
        return y_unit, page_height

    header_rules = [rules_below[0]]
    for r in rules_below[1:]:
        if r - header_rules[-1] <= _HEADER_RULE_GAP:
            header_rules.append(r)
        else:
            break

    data_start = header_rules[-1]
    remaining = [r for r in rules_below if r > data_start]
    section_end = remaining[0] if remaining else page_height

    return data_start, section_end


def build_unit_sections(lines: list[Line], page: pdfplumber.page.Page) -> list[UnitSection]:
    """Find all unit headers on *page* and compute the section each one owns."""
    rules = get_horizontal_rules(page)
    sections: list[UnitSection] = []
    for line in lines:
        factor = _parse_unit_from_text(line.full_text)
        if factor is not None:
            data_start, section_end = _find_unit_scope(rules, line.y, page.height)
            sections.append(UnitSection(
                y_header=line.y,
                factor=factor,
                evidence=f"scoped unit header: {line.full_text!r}",
                data_start=data_start,
                section_end=section_end,
            ))
    return sections


def find_scoped_unit_context(sections: list[UnitSection], y: float) -> Multiplier | None:
    """Return the unit Multiplier if *y* falls within any UnitSection's scope.

    A number is in scope if it is on the header line itself or between
    data_start and section_end.
    """
    for sec in sections:
        on_header = abs(y - sec.y_header) < 2
        in_section = sec.data_start < y < sec.section_end
        if on_header or in_section:
            return Multiplier(factor=sec.factor, evidence=sec.evidence)
    return None


_YEAR_PREFIX_RE = re.compile(r"^(FY|CY)\d{2,4}$", re.IGNORECASE)


def _is_pure_numeric_token(token: Token, prev_token: Token | None) -> bool:
    """Return True if *token* looks like a data number rather than a label.

    Fiscal/calendar year labels like 'FY2025' (single token) or '2025' preceded
    by an 'FY'/'CY' token are treated as labels, not data numbers.
    """
    stripped = token.text.replace(",", "").replace(".", "").replace("-", "")
    if not stripped.isdigit():
        return False
    if _YEAR_PREFIX_RE.match(token.text):
        return False
    if prev_token is not None and prev_token.text.upper() in ("FY", "CY"):
        return False
    return True


def _is_header_like(line: Line) -> bool:
    """Return True if the line looks like a column header rather than data.

    Heuristic: fewer than 30% of tokens parse as pure data numbers.
    A header line mostly contains labels; a data line mostly contains numbers.
    """
    if not line.tokens:
        return False
    numeric_count = sum(
        1 for i, t in enumerate(line.tokens)
        if _is_pure_numeric_token(t, line.tokens[i - 1] if i > 0 else None)
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
    unit_sections = build_unit_sections(lines, page)

    results: list[tuple[NumberMatch, Multiplier]] = []
    for nm in numbers:
        col_header = find_column_header(lines, nm.y, nm.x0, nm.x1)
        unit_ctx = find_scoped_unit_context(unit_sections, nm.y)
        inline = detect_inline_multiplier(page_text, nm.position)
        mult = resolve_multiplier(col_header, unit_ctx, inline)
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
