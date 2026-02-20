from __future__ import annotations

from dataclasses import dataclass


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
    position: int
    page_number: int
    x0: float
    x1: float
    y: float


@dataclass
class Multiplier:
    """Unit multiplier derived from surrounding context."""

    factor: float
    evidence: str


@dataclass
class UnitSection:
    """A unit header and the y-range it owns on a page."""

    y_header: float
    factor: float
    evidence: str
    data_start: float
    section_end: float
