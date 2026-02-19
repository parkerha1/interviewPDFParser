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

import argparse
import heapq
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


def extract_numbers(
    lines: list[Line],
    page_number: int,
    page_text: str,
    include_negatives: bool = False,
) -> list[NumberMatch]:
    """Return numbers found in *lines*.

    By default only positive numbers are returned. When *include_negatives* is
    True, accounting-notation negatives like ``(364.7)`` are included with
    negative values.
    """
    negative_spans: dict[tuple[int, int], str] = {}
    for m in _NEGATIVE_RE.finditer(page_text):
        negative_spans[(m.start(), m.end())] = m.group(1)

    results: list[NumberMatch] = []

    for line in lines:
        for token in line.tokens:
            raw = token.text

            for m in _NUMBER_RE.finditer(raw):
                token_raw = m.group()
                if token_raw.replace(",", "").replace(".", "") == "":
                    continue

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

                position = page_text.find(token_raw)

                is_negative = any(
                    position >= ns and position + len(token_raw) <= ne
                    for ns, ne in negative_spans
                )
                if is_negative and not include_negatives:
                    continue

                results.append(NumberMatch(
                    value=-value if is_negative else value,
                    raw_text=f"({token_raw})" if is_negative else token_raw,
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


_LABEL_COL_THRESHOLD = 0.5  # a column is "label" if >50% of overlapping tokens are text


def _is_token_textual(token: Token) -> bool:
    """Return True if *token* is predominantly text rather than a number."""
    stripped = token.text.replace(",", "").replace(".", "").replace("-", "").replace("(", "").replace(")", "")
    if not stripped:
        return True
    return not stripped.isdigit()


def is_in_label_column(lines: list[Line], x0: float, x1: float, _cache: dict | None = None) -> bool:
    """Return True if the vertical slice at [x0, x1] is predominantly text.

    Scans all tokens on all lines that overlap the x-range and checks whether
    the majority are text rather than numbers. A cache keyed by rounded x0
    avoids redundant work for numbers at the same column position.
    """
    if _cache is not None:
        key = round(x0, 0)
        if key in _cache:
            return _cache[key]

    text_count = 0
    total = 0
    for line in lines:
        for token in line.tokens:
            if token.x1 >= x0 and token.x0 <= x1:
                total += 1
                if _is_token_textual(token):
                    text_count += 1

    result = (text_count / total > _LABEL_COL_THRESHOLD) if total > 0 else False

    if _cache is not None:
        _cache[round(x0, 0)] = result
    return result


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

def process_page(
    page: pdfplumber.page.Page,
    include_negatives: bool = False,
) -> list[tuple[NumberMatch, Multiplier]]:
    """Extract all numbers from *page* and resolve each one's unit multiplier."""
    page_text = page.extract_text() or ""
    if not page_text.strip():
        return []

    lines = chars_to_lines(page.chars)
    numbers = extract_numbers(lines, page.page_number, page_text, include_negatives)
    unit_sections = build_unit_sections(lines, page)

    label_col_cache: dict[float, bool] = {}
    results: list[tuple[NumberMatch, Multiplier]] = []
    for nm in numbers:
        if is_in_label_column(lines, nm.x0, nm.x1, label_col_cache):
            results.append((nm, Multiplier(factor=1, evidence="label column")))
            continue

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


def _print_entry(
    rank: int,
    label: str,
    nm: NumberMatch,
    mult: Multiplier,
    adjusted: float,
    page_texts: dict[int, str],
    verbose: bool,
) -> None:
    """Print one result entry."""
    prefix = f"  #{rank}" if rank else " "
    print(f"{prefix} {label}: {adjusted:>24,.2f}")
    print(f"      Raw text:    {nm.raw_text!r}  (page {nm.page_number})")
    if verbose:
        print(f"      Multiplier:  {mult.factor:,.0f}x  ({mult.evidence})")
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

            for nm, mult in process_page(page, include_negatives):
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
    for i, (val, _, nm) in enumerate(raw_sorted, 1):
        no_mult = Multiplier(factor=1, evidence="raw")
        _print_entry(i if show_rank else 0, "Raw", nm, no_mult, nm.value,
                     page_texts, verbose)

    print(f"\nLargest adjusted number{'s' if top_n > 1 else ''}{neg_note}:\n")
    for i, (adj_val, _, nm, mult) in enumerate(adj_sorted, 1):
        _print_entry(i if show_rank else 0, "Adjusted", nm, mult, adj_val,
                     page_texts, verbose)

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find the largest numerical value in a PDF document.",
    )
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument(
        "-n", "--top-n",
        type=int, default=1, metavar="N",
        help="Show the top N largest numbers (default: 1)",
    )
    parser.add_argument(
        "--include-negatives",
        action="store_true",
        help="Include accounting-notation negatives like (364.7)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show multiplier evidence and context for each result",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    find_largest(
        args.pdf,
        top_n=args.top_n,
        include_negatives=args.include_negatives,
        verbose=args.verbose,
    )
