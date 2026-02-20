"""Find the largest numerical value in a PDF document."""

from __future__ import annotations

import argparse
from pathlib import Path

from pdf_pipeline import find_largest

_DEFAULT_PDF = Path(__file__).parent / "FY25 Air Force Working Capital Fund.pdf"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find the largest numerical value in a PDF document.",
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        default=str(_DEFAULT_PDF),
        help=f"Path to the PDF file (default: {_DEFAULT_PDF.name!r})",
    )
    parser.add_argument(
        "-n",
        "--top-n",
        type=int,
        default=1,
        metavar="N",
        help="Show the top N largest numbers (default: 1)",
    )
    parser.add_argument(
        "--include-negatives",
        action="store_true",
        help="Include accounting-notation negatives like (364.7)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
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
