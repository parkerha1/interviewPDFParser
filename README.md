# PDF Largest Number Finder

Finds the largest numerical value in a PDF document, both as a raw number and adjusted by natural-language unit context (e.g. "Dollars in Millions").

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python solution.py
```

The included PDF (`FY25 Air Force Working Capital Fund.pdf`) is used by default. The script outputs the largest raw number and the largest context-adjusted number found, along with the page and surrounding text for verification.

### Options

```
-n N, --top-n N        Show the top N largest numbers (default: 1)
--include-negatives    Include accounting-notation negatives like (364.7),
                       ranked by absolute magnitude
-v, --verbose          Show multiplier evidence and context for each result
```

### Examples

```bash
# Top 5 largest, with full evidence trail
python solution.py -n 5 -v

# Include negatives, show top 10
python solution.py -n 10 --include-negatives

# Run against a different PDF
python solution.py path/to/other.pdf -n 5 -v
```

## Notes

- Activate the virtualenv (`source .venv/bin/activate`) each time you open a new terminal before running the script.
- To deactivate the virtualenv when done: `deactivate`
