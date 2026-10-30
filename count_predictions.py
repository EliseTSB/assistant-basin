"""Count the rows of PREDICTIONS.md by verdict, for the README's front-page line.

Rows in tables with a Verdict or Status header are read from that column. Other tables use the
old rule: the verdict is read from the start of the cells after the prediction text. Rows whose
verdict the patterns below cannot read are classified by hand in OVERRIDES; a new unreadable
row is printed as UNCLASSIFIED and the script exits 1, so a count is never silently wrong.

Calibration: on the PREDICTIONS.md of commit 09a99a3, where the README first said "230 predictions so far: 132 held,
71 failed", this reads 230 rows, 133 held and 76 failed. The total matches; the old hand count was made another way.

Usage: python3 count_predictions.py [PREDICTIONS.md]
"""
import collections
import re
import sys

OVERRIDES = {
    '11.2': 'mixed', '34.1': 'mixed', '34.6': 'mixed',
    '34.2': 'failed', '34.12': 'failed', '64e': 'failed',
    '20g.2': 'untestable', '25.8': 'untestable', '25.9': 'untestable', '34.7': 'untestable',
    '50.3': 'untestable', '50.4': 'untestable', '17b.4': 'untestable',
    '32.2': 'reading', '33.1': 'reading',
    '43.3a': 'unscored',
    # Checked by hand 2026-10-29. Gated or not tested, like 17b.4: 75.3 (75.2 failed), 80.5 and 80.6 (80.1 failed),
    # 99.1 and 99.2 (Base answers loop in the chat frame). Part 89's table names its verdict column "evidence".
    '75.3': 'untestable', '80.5': 'untestable', '80.6': 'untestable', '99.1': 'untestable', '99.2': 'untestable',
    '89.1': 'held', '89.2': 'failed', '89.3': 'failed', '89.4': 'failed', '89.5': 'failed', '89.6': 'failed',
    '89.7': 'held',
    # 86.5: the top-20 verification it needs was never run (Part 86 README).
    '86.5': 'untestable',
}
PATTERNS = [
    ('unscored', r'^(pending|not run|not scored|unscored|running|not yet|—|-|)$|^(pending|not run|not yet|unscored)'),
    ('mixed', r'^(mixed|partly|partial)'),
    ('held', r'^(held|holds|confirmed|passed|pass\b|yes\b)'),
    ('failed', r'^(failed|fails|missed|refuted|no\b|did not)'),
]
ROW_ID = re.compile(r'\d+[a-z]?(\.\d+[a-z]?)*|\d+[a-z]+\d*')


def _split_cells(line, respect_escaped_pipes=False):
    content = line.strip().strip('|')
    if not respect_escaped_pipes:
        return [cell.strip() for cell in content.split('|')]

    cells, current = [], []
    backslashes = 0
    for character in content:
        if character == '|' and backslashes % 2 == 0:
            cells.append(''.join(current).strip())
            current = []
            backslashes = 0
            continue
        current.append(character)
        if character == '\\':
            backslashes += 1
        else:
            backslashes = 0
    cells.append(''.join(current).strip())
    return cells


def _header_name(cell):
    """Normalize a Markdown-formatted table heading."""
    return re.sub(r'[*_`]', '', cell).strip().casefold()


def _is_separator(cells):
    return bool(cells) and all(
        re.fullmatch(r':?-{3,}:?', re.sub(r'\s+', '', cell)) for cell in cells
    )


def _looks_like_header(cells):
    names = [_header_name(cell) for cell in cells]
    return any(name in {'#', 'id', 'prediction'} for name in names)


def rows(text):
    """Yield each prediction id and the text used to read its verdict."""
    verdict_column = None
    for line in text.split('\n'):
        if not line.startswith('|'):
            verdict_column = None
            continue
        cells = _split_cells(line, respect_escaped_pipes=True)
        if not cells:
            continue
        if ROW_ID.fullmatch(cells[0]):
            if verdict_column is not None:
                verdict_text = cells[verdict_column] if verdict_column < len(cells) else ''
            else:
                # Preserve the historical rule for tables without a verdict/status heading.
                verdict_text = ' '.join(cells[2:])
            yield cells[0], verdict_text.lower()
            continue

        if _is_separator(cells):
            continue

        names = [_header_name(cell) for cell in cells]
        if 'verdict' in names:
            verdict_column = names.index('verdict')
        elif 'status' in names:
            verdict_column = names.index('status')
        elif _looks_like_header(cells):
            # A new table without a verdict/status heading must use the historical rule.
            verdict_column = None


def verdict(row_id, text):
    if row_id in OVERRIDES:
        return OVERRIDES[row_id]
    head = re.sub(r'[*_`]', '', text)[:60].strip()
    for name, pattern in PATTERNS:
        if re.search(pattern, head):
            return name
    return None


def classify(text):
    counts, unread = collections.Counter(), []
    for row_id, row_text in rows(text):
        value = verdict(row_id, row_text)
        if value is None:
            unread.append((row_id, row_text[:90]))
        else:
            counts[value] += 1
    return counts, unread


def main(text=None):
    path = sys.argv[1] if len(sys.argv) > 1 else 'PREDICTIONS.md'
    if text is None:
        with open(path, encoding='utf-8') as prediction_file:
            text = prediction_file.read()
    counts, unread = classify(text)
    total = sum(counts.values()) + len(unread)
    print(f"{total} rows: " + ', '.join(f"{key} {counts[key]}" for key in
          ('held', 'failed', 'mixed', 'untestable', 'reading', 'unscored')))
    for row_id, row_text in unread:
        print('UNCLASSIFIED', row_id, '|', row_text)
    return 1 if unread else 0


if __name__ == '__main__':
    sys.exit(main())
