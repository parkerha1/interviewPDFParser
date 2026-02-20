from __future__ import annotations

import re

import pdfplumber

from pdf_extract import chars_to_lines, extract_numbers
from pdf_models import Line, Multiplier, NumberMatch, Token, UnitSection

_UNIT_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(r"\bin\s+billions?\b", re.IGNORECASE), 1_000_000_000),
    (re.compile(r"\bin\s+millions?\b", re.IGNORECASE), 1_000_000),
    (re.compile(r"\bin\s+thousands?\b", re.IGNORECASE), 1_000),
]

_PER_RE = re.compile(r"\bper\b", re.IGNORECASE)
_INLINE_UNIT_RE = re.compile(r"\b(billion|million|thousand)s?\b", re.IGNORECASE)
_INLINE_FACTORS = {"billion": 1_000_000_000, "million": 1_000_000, "thousand": 1_000}

_HEADER_RULE_GAP = 30
_INLINE_SCAN_WINDOW = 60
_COL_HEADER_TOKEN_GAP = 20
_LABEL_COL_THRESHOLD = 0.5
_YEAR_PREFIX_RE = re.compile(r"^(FY|CY)\d{2,4}$", re.IGNORECASE)


def _parse_unit_from_text(text: str) -> float | None:
    for pattern, factor in _UNIT_PATTERNS:
        if pattern.search(text):
            return factor
    return None


def get_horizontal_rules(page: pdfplumber.page.Page, min_width_frac: float = 0.3) -> list[float]:
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


def build_unit_sections(lines: list[Line], rules: list[float], page_height: float) -> list[UnitSection]:
    sections: list[UnitSection] = []
    for line in lines:
        factor = _parse_unit_from_text(line.full_text)
        if factor is not None:
            data_start, section_end = _find_unit_scope(rules, line.y, page_height)
            sections.append(
                UnitSection(
                    y_header=line.y,
                    factor=factor,
                    evidence=f"scoped unit header: {line.full_text!r}",
                    data_start=data_start,
                    section_end=section_end,
                )
            )
    return sections


def find_scoped_unit_context(sections: list[UnitSection], y: float) -> Multiplier | None:
    for sec in sections:
        on_header = abs(y - sec.y_header) < 2
        in_section = sec.data_start < y < sec.section_end
        if on_header or in_section:
            return Multiplier(factor=sec.factor, evidence=sec.evidence)
    return None


def _is_token_textual(token: Token) -> bool:
    stripped = (
        token.text.replace(",", "")
        .replace(".", "")
        .replace("-", "")
        .replace("(", "")
        .replace(")", "")
    )
    if not stripped:
        return True
    return not stripped.isdigit()


def _find_table_y_bounds(rules: list[float], y: float, page_height: float) -> tuple[float, float]:
    rule_above = max((r for r in rules if r <= y), default=0.0)
    rule_below = min((r for r in rules if r > y), default=page_height)
    return rule_above, rule_below


def is_in_label_column(
    lines: list[Line],
    x0: float,
    x1: float,
    y: float,
    rules: list[float],
    page_height: float,
    _cache: dict | None = None,
) -> bool:
    y_top, y_bot = _find_table_y_bounds(rules, y, page_height)

    if _cache is not None:
        key = (round(x0, 0), y_top, y_bot)
        if key in _cache:
            return _cache[key]

    text_count = 0
    total = 0
    for line in lines:
        if line.y < y_top or line.y > y_bot:
            continue
        for token in line.tokens:
            if token.x1 >= x0 and token.x0 <= x1:
                total += 1
                if _is_token_textual(token):
                    text_count += 1

    result = (text_count / total > _LABEL_COL_THRESHOLD) if total > 0 else False

    if _cache is not None:
        _cache[(round(x0, 0), y_top, y_bot)] = result
    return result


def _is_pure_numeric_token(token: Token, prev_token: Token | None) -> bool:
    stripped = token.text.replace(",", "").replace(".", "").replace("-", "")
    if not stripped.isdigit():
        return False
    if _YEAR_PREFIX_RE.match(token.text):
        return False
    if prev_token is not None and prev_token.text.upper() in ("FY", "CY"):
        return False
    return True


def _is_header_like(line: Line) -> bool:
    if not line.tokens:
        return False
    numeric_count = sum(
        1
        for i, t in enumerate(line.tokens)
        if _is_pure_numeric_token(t, line.tokens[i - 1] if i > 0 else None)
    )
    return numeric_count / len(line.tokens) < 0.3


def find_column_header(lines: list[Line], y: float, x0: float, x1: float) -> str | None:
    above = [ln for ln in lines if ln.y < y]
    for ln in reversed(above):
        if not _is_header_like(ln):
            continue

        match_idx = None
        for i, token in enumerate(ln.tokens):
            if token.x1 >= x0 and token.x0 <= x1:
                match_idx = i
                break

        if match_idx is None:
            continue

        phrase_tokens = [ln.tokens[match_idx]]
        i = match_idx - 1
        while i >= 0:
            gap = phrase_tokens[0].x0 - ln.tokens[i].x1
            if gap <= _COL_HEADER_TOKEN_GAP:
                phrase_tokens.insert(0, ln.tokens[i])
                i -= 1
            else:
                break

        i = match_idx + 1
        while i < len(ln.tokens):
            gap = ln.tokens[i].x0 - phrase_tokens[-1].x1
            if gap <= _COL_HEADER_TOKEN_GAP:
                phrase_tokens.append(ln.tokens[i])
                i += 1
            else:
                break

        return " ".join(t.text for t in phrase_tokens)

    return None


def find_same_row_unit(
    lines: list[Line],
    y: float,
    x0: float,
    rules: list[float],
    page_height: float,
    label_col_cache: dict[tuple[float, float, float], bool] | None = None,
) -> Multiplier | None:
    for line in lines:
        if abs(line.y - y) > 3:
            continue
        first_token = line.tokens[0] if line.tokens else None
        if first_token is None or first_token.x0 >= x0:
            continue
        if not is_in_label_column(
            lines, first_token.x0, first_token.x1, y, rules, page_height, label_col_cache
        ):
            continue
        left_text = " ".join(t.text for t in line.tokens if t.x1 <= x0)
        factor = _parse_unit_from_text(left_text)
        if factor is not None:
            return Multiplier(factor=factor, evidence=f"same-row label: {left_text!r}")
    return None


def detect_inline_multiplier(page_text: str, position: int) -> Multiplier | None:
    window = page_text[position : position + _INLINE_SCAN_WINDOW]
    m = _INLINE_UNIT_RE.search(window)
    if m:
        factor = _INLINE_FACTORS[m.group(1).lower()]
        return Multiplier(factor=factor, evidence=f"inline: {window.strip()!r}")
    return None


def resolve_multiplier(
    col_header: str | None,
    unit_ctx: Multiplier | None,
    inline_mult: Multiplier | None,
    same_row: Multiplier | None = None,
) -> Multiplier:
    if col_header is not None:
        if _PER_RE.search(col_header):
            return Multiplier(factor=1, evidence=f"rate column: {col_header!r}")

        factor = _parse_unit_from_text(col_header)
        if factor is not None:
            return Multiplier(factor=factor, evidence=f"column header: {col_header!r}")

    if inline_mult is not None:
        return inline_mult

    if same_row is not None:
        return same_row

    if col_header is not None and unit_ctx is not None:
        return unit_ctx

    return Multiplier(factor=1, evidence="no context found")


def process_page(
    page: pdfplumber.page.Page,
    include_negatives: bool = False,
    page_text: str | None = None,
) -> list[tuple[NumberMatch, Multiplier]]:
    if page_text is None:
        page_text = page.extract_text() or ""
    if not page_text.strip():
        return []

    lines = chars_to_lines(page.chars)
    numbers = extract_numbers(lines, page.page_number, page_text, include_negatives)
    rules = get_horizontal_rules(page)
    unit_sections = build_unit_sections(lines, rules, page.height)

    label_col_cache: dict[tuple[float, float, float], bool] = {}
    results: list[tuple[NumberMatch, Multiplier]] = []
    for nm in numbers:
        if is_in_label_column(lines, nm.x0, nm.x1, nm.y, rules, page.height, label_col_cache):
            results.append((nm, Multiplier(factor=1, evidence="label column")))
            continue

        col_header = find_column_header(lines, nm.y, nm.x0, nm.x1)
        unit_ctx = find_scoped_unit_context(unit_sections, nm.y)
        inline = detect_inline_multiplier(page_text, nm.position)
        same_row = find_same_row_unit(lines, nm.y, nm.x0, rules, page.height, label_col_cache)
        mult = resolve_multiplier(col_header, unit_ctx, inline, same_row)
        results.append((nm, mult))

    return results
