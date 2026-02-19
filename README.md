# PDF Largest Number Finder

Finds the largest numerical value in a PDF document, both as a raw number and adjusted by natural-language unit context (e.g. "Dollars in Millions").

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python solution.py "FY25 Air Force Working Capital Fund.pdf"
```

Pass any PDF path as the argument. The script outputs the largest raw number and the largest context-adjusted number found, along with the page and surrounding text for verification.
