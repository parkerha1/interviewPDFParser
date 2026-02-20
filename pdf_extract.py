from __future__ import annotations

import re
from collections import defaultdict

from pdf_models import Line, NumberMatch, Token


def chars_to_lines(chars: list[dict]) -> list[Line]:
    """Group page.chars by y-coordinate into sorted Line objects."""
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
            avg_char_width = (current_x1 - current_x0) / len(current_text) if current_text else 5.0
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


_NUMBER_RE = re.compile(r"[\d,]+\.?\d*")
_NEGATIVE_RE = re.compile(r"\(\s*([\d,]+\.?\d*)\s*\)")
_UNIT_SUFFIXES = frozenset("MKBTmkbt")


def extract_numbers(
    lines: list[Line],
    page_number: int,
    page_text: str,
    include_negatives: bool = False,
) -> list[NumberMatch]:
    """Return numbers found in *lines*."""
    negative_spans: list[tuple[int, int]] = [
        (m.start(), m.end()) for m in _NEGATIVE_RE.finditer(page_text)
    ]

    results: list[NumberMatch] = []
    search_offset = 0

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

                position = page_text.find(token_raw, search_offset)
                if position == -1:
                    position = page_text.find(token_raw)
                else:
                    search_offset = position + len(token_raw)

                is_negative = any(
                    position >= ns and position + len(token_raw) <= ne
                    for ns, ne in negative_spans
                )
                if is_negative and not include_negatives:
                    continue

                results.append(
                    NumberMatch(
                        value=-value if is_negative else value,
                        raw_text=f"({token_raw})" if is_negative else token_raw,
                        position=position,
                        page_number=page_number,
                        x0=token.x0,
                        x1=token.x1,
                        y=line.y,
                    )
                )

    return results
