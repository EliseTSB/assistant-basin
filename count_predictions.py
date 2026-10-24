"""Count the rows of PREDICTIONS.md by verdict, for the README's front-page line.

A prediction row is a table row whose first cell is an id such as 1.2, 12c.3 or 65a. Its verdict is read from the start
of the cells after the prediction text (the Status column where a table has one, else the Result). Rows whose verdict
the patterns below cannot read are classified by hand in OVERRIDES (checked 2026-10-24); a new unreadable row is
printed as UNCLASSIFIED and the script exits 1, so the count is never silently wrong.

Calibration: on the PREDICTIONS.md of commit 09a99a3, where the README first said "230 predictions so far: 132 held,
71 failed", this reads 230 rows, 133 held and 76 failed. The total matches; the old hand count was made another way.

Usage: python3 count_predictions.py [PREDICTIONS.md]
"""
import collections, re, sys

OVERRIDES = {
    '11.2': 'mixed', '34.1': 'mixed', '34.6': 'mixed',
    '34.2': 'failed', '34.12': 'failed', '64e': 'failed',
    '20g.2': 'untestable', '25.8': 'untestable', '25.9': 'untestable', '34.7': 'untestable',
    '50.3': 'untestable', '50.4': 'untestable', '17b.4': 'untestable',
    '32.2': 'reading', '33.1': 'reading',
    '43.3a': 'unscored',
}
PATTERNS = [
    ('unscored', r'^(pending|not run|not scored|unscored|running|not yet|—|-|)$|^(pending|not run|not yet|unscored)'),
    ('mixed', r'^(mixed|partly|partial)'),
    ('held', r'^(held|holds|confirmed|passed|pass\b|yes\b)'),
    ('failed', r'^(failed|fails|missed|refuted|no\b|did not)'),
]


def rows(text):
    for line in text.split('\n'):
        if not line.startswith('|'):
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if not cells or not re.fullmatch(r'\d+[a-z]?(\.\d+[a-z]?)*|\d+[a-z]+\d*', cells[0]):
            continue
        yield cells[0], ' '.join(cells[2:]).lower()


def verdict(row_id, text):
    if row_id in OVERRIDES:
        return OVERRIDES[row_id]
    head = re.sub(r'[*_`]', '', text)[:60].strip()
    for name, pattern in PATTERNS:
        if re.search(pattern, head):
            return name
    return None


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'PREDICTIONS.md'
    counts, unread = collections.Counter(), []
    for row_id, text in rows(open(path, encoding='utf-8').read()):
        v = verdict(row_id, text)
        if v is None:
            unread.append((row_id, text[:90]))
        else:
            counts[v] += 1
    total = sum(counts.values()) + len(unread)
    print(f"{total} rows: " + ', '.join(f"{k} {counts[k]}" for k in
          ('held', 'failed', 'mixed', 'untestable', 'reading', 'unscored')))
    for row_id, text in unread:
        print('UNCLASSIFIED', row_id, '|', text)
    return 1 if unread else 0


if __name__ == '__main__':
    sys.exit(main())
