# Progress Log

## Problem

Find the largest numerical value in `FY25 Air Force Working Capital Fund.pdf` — both as a raw number and as a context-adjusted number where natural language guidance (e.g. "Dollars in Millions") is applied as a multiplier.

## Document Characteristics

- 114 pages, ~13MB PDF
- Mostly financial budget tables with a page-level unit header near the top of each page
- Common unit headers: `(Dollars in Millions)`, `(Dollars in Thousands)`, `(Hours in Thousands)`
- Numbers use comma-grouped formatting: `28,239.2`
- Negative numbers use accounting notation: `(364.7)` — parentheses, not minus sign
- `pdfplumber.find_tables()` only detects bordered tables on 45 of 114 pages; the remaining 69 pages have whitespace-aligned tabular data that isn't detected as a table
- Many pages have mixed-unit tables: a page header says "Dollars in Millions" but individual columns like `$ Per Barrel` or `Sales Rate per hour` are rates with no multiplier

## Current Results

```
Largest raw number:      6,000,000
  Raw text:              '6,000,000'
  Page:                  93
  Context:               ...projects are smaller in scale (costing between $250,000 and $6,000,000)...

Largest adjusted number: 35,110,000,000
  Raw text:              '35,110'
  Multiplier:            1,000,000x (unit header: '(Dollars in Millions) FY 2023 FY 2024 FY 2025')
  Page:                  13
  Context:               ...Civilian End Strength 33,848 34,735 35,110 Military End Strength...
```

## Architecture

Single file: `solution.py`. Dependency: `pdfplumber` (pip install).

### Pipeline overview

```
page.chars
  → chars_to_lines()        # group chars by y-coord into Line/Token objects with x0/x1
  → extract_numbers()       # regex over tokens, yield NumberMatch with x0/x1/y coords
  → process_page()          # for each number: find_column_header + find_unit_context + detect_inline_multiplier
                            # → resolve_multiplier() applies specificity hierarchy
  → find_largest()          # track raw_max and adj_max across all pages
```

### Key design decisions

**Why `page.chars` instead of `extract_text()`**
`extract_text()` gives a flat string. Numbers from adjacent columns sometimes merge into one token (e.g. `40` and `314.090` becoming `40314.090`) because columns are positioned by coordinate in the PDF, not by explicit space characters. Working directly with `page.chars` gives real x/y coordinates for each character.

**Gap-based token splitting in `chars_to_lines`**
Tokens are split on both explicit space characters AND x-coordinate gaps wider than 1.5× the average character width of the current token (minimum 4px). This correctly separates columns that have no space characters between them.

**"Always look upward" strategy**
Rather than detecting table regions, every number scans upward from its y-position for context. This is more generalizable: a page with two tables with different units works correctly because each number finds the header above its own table, not a page-level scan.

**Two independent upward scans per number**
1. `find_column_header(lines, y, x0, x1)` — finds the nearest header-like line above y, checks x-alignment
2. `find_unit_context(lines, y)` — finds the nearest line containing a unit keyword (in millions/thousands/billions)

**Specificity hierarchy in `resolve_multiplier`**
1. Column header containing `per` (word boundary) → rate column, multiplier = 1
2. Column header containing a unit keyword → use that unit
3. Inline cue immediately after the number (e.g. `$3.2 million`)
4. Column header found (any) + unit context above → use unit context
   - **Critical:** `unit_ctx` is only applied when a column header was also found. Prose numbers (no column header above them) do NOT inherit a unit from an unrelated table elsewhere on the page.
5. No cue → multiplier = 1

**Multi-word column phrase reconstruction**
`find_column_header` finds the overlapping token but also expands left/right to adjacent tokens within 20px to reconstruct phrases like `$ Per Barrel`. Without this, a number under the `Barrel` token would not see the `Per` word in the same phrase.

**`_is_header_like` heuristic**
A line is considered a column header if fewer than 30% of its tokens parse as pure numbers. Header lines are mostly labels; data lines are mostly numbers.

## Key Discoveries During Development

### `find_tables()` misses most pages
Only 45/114 pages have detected tables. The major financial table pages (30–40, 56–66, 78–82) use whitespace-aligned layout with no borders. This ruled out using `find_tables()` as the primary approach.

### Camelot considered and rejected
Camelot's `stream` mode handles borderless tables but requires Ghostscript as a system dependency (not a pip install). The prompt asks for a self-contained solution. `page.chars` achieves the same result without the dependency.

### Column merging bug
Before gap-based splitting, numbers like `40` (quantity) and `314.090` (cost) on the same row with no space character were merging into `40314.090`. Fixed by splitting on x-coordinate gaps.

### `$ Per Barrel` layout (pages 38–40)
The column header is `PRODUCT Barrels $ Per Barrel TOTAL`. Numbers in the `$ Per Barrel` column align spatially under the `Barrel` token, not the `Per` token. Single-token matching returned `Barrel`, which doesn't trigger the "per" rule. Fixed by expanding the returned phrase to include adjacent tokens within 20px.

### Accounting negatives
Parenthesized negatives like `(364.7)` are excluded. We first collect all negative spans with `_NEGATIVE_RE`, then skip any number whose flat-text position falls inside one.

### Alphanumeric code filtering
Digit runs adjacent to letters are usually identifier codes (e.g. `0708055F`). These are filtered, except for unit-suffix letters `M/K/B/T` which are preserved for context resolution to handle later.

## Known Problems / Edge Cases

### Semantic mismatch on page 13 (current adjusted maximum)
The current adjusted maximum is `35,110 × 1,000,000 = 35,110,000,000` from "Civilian End Strength" on page 13. The page header says `(Dollars in Millions)` but end-strength figures are people counts, not dollars. The system correctly applies the stated unit context mechanically, but the semantic meaning is wrong. Fixing this would require understanding row semantics (personnel vs financial).

### Page 13 financial numbers not getting multiplier
Counterintuitively, the large financial totals on page 13 (`28,239.2`, `29,176.6`, `30,704.1`) receive no multiplier ("no context found"), while the headcount rows below them do. Root cause: the column header line `FY 2023 FY 2024 FY 2025` has tokens `FY`, `2023`, `FY`, `2024`, `FY`, `2025` — 33% numeric — which fails `_is_header_like`'s < 30% threshold and is skipped. The financial numbers have no column header found, so `unit_ctx` is also skipped (prose rule). The headcount numbers happen to have a different header-like line above them that passes the threshold.

**Potential fix:** Tune the `_is_header_like` threshold, or treat `FYxxxx` tokens as non-numeric since they contain letters.

### Page 27 rate row not suppressed
`Sales Rate per hour` values (`339.5`, `381.1`, `425.5`) are not being suppressed with multiplier=1. The phrase is a row label next to the numbers, not a column header above them. `find_column_header` scans upward and finds no x-aligned header-like line containing "per". These values are small and don't affect the final result.

**Potential fix:** Also scan the same y-line (to the left) for "per" keywords, not just upward.

### `position` field for inline scan is fragile
`extract_numbers` computes `position` as `page_text.find(token_raw)` — a substring search in the flat text. If the same number string appears multiple times on the page (e.g. `0.000`), `find()` always returns the first occurrence, which may be the wrong position. The `detect_inline_multiplier` window would then look at the wrong part of the text.

**Potential fix:** Build a positional map from char coordinates back to flat-text offsets rather than using `str.find()`.

### Unit keywords must be "in X" form
`find_unit_context` only matches `in millions`, `in thousands`, `in billions`. It would miss patterns like `expressed in millions` (which does appear in the document, page 27 prose) or `values are in thousands`. The regex could be broadened.

### `_UNIT_SUFFIXES` not yet acted upon
The extraction preserves numbers adjacent to `M/K/B/T` suffixes (e.g. `$4.2M`) for context resolution to handle. But `resolve_multiplier` doesn't currently check for these suffixes. Inline patterns like `4.2M` → 4,200,000 are not yet resolved.

## Next Steps

1. Fix `_is_header_like` to treat `FYxxxx` tokens as non-numeric (fixes page 13 financial numbers)
2. Implement suffix multiplier resolution (`4.2M` → 4,200,000) in `resolve_multiplier` or `detect_inline_multiplier`
3. Fix the `position` field to use coordinate-based flat-text mapping instead of `str.find()`
4. Consider scanning left on the same line for "per" to handle row-label rate suppression (page 27)
5. Broaden `_UNIT_PATTERNS` to catch more phrasings of unit context
