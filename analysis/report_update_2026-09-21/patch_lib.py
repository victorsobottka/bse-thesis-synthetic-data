"""Small helpers for the template rewrite: exact, unique replacements in generate_report.py (an edit that does not match exactly once fails loudly)."""
from pathlib import Path
GR = Path(__file__).resolve().parents[2] / 'generate_report.py'


def load():
    return GR.read_text()


def save(t):
    GR.write_text(t)


def replace_once(t, old, new):
    n = t.count(old)
    assert n == 1, f"expected exactly one match, found {n}: {old[:90]!r}"
    return t.replace(old, new)


def replace_line(t, prefix, new):
    """Replace the single line that starts with `prefix`."""
    lines = t.split('\n')
    idx = [i for i, l in enumerate(lines) if l.startswith(prefix)]
    assert len(idx) == 1, f"expected exactly one line starting with {prefix[:70]!r}, found {len(idx)}"
    lines[idx[0]] = new
    return '\n'.join(lines)


def replace_between(t, start, end, new):
    """Replace from `start` (inclusive) up to `end` (exclusive); both must occur exactly once, start before end."""
    assert t.count(start) == 1, f"start marker count {t.count(start)}: {start[:70]!r}"
    i = t.index(start)
    j = t.index(end, i + len(start))
    assert t.count(end, i + len(start)) >= 1
    return t[:i] + new + t[j:]


def insert_before(t, marker, new):
    assert t.count(marker) == 1, f"marker count {t.count(marker)}: {marker[:70]!r}"
    return t.replace(marker, new + marker)


def insert_after(t, marker, new):
    assert t.count(marker) == 1, f"marker count {t.count(marker)}: {marker[:70]!r}"
    return t.replace(marker, marker + new)
