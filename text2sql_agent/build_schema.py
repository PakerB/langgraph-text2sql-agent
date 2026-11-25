"""Build an enriched schema text for each BIRD database (run once).

For every column: type, primary/foreign keys (from SQLite), full name and description
(from BIRD's database_description CSVs), stored values (all values if there are at most
10 distinct ones, otherwise the 3 most frequent) and the meaning of values.
Databases and CSV files are only read.

Usage:  python -m text2sql_agent.build_schema      -> schemas/<db_id>.txt
"""
import csv
import io
import re
import sqlite3

from .common import DB_ROOT, RICH_SCHEMA_DIR, db_path

MAX_LIST_ALL = 10
N_EXAMPLES = 3
MAX_VALUE_CHARS = 60


def read_csv_any_encoding(path):
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):   # a few BIRD description files are cp1252
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    return [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(io.StringIO(text))]


def clean(text: str) -> str:
    """Collapse whitespace and drop the 'commonsense evidence:' label."""
    text = re.sub(r"(?i)commonsense (evidence|reasoning)\s*:", "", text)
    return re.sub(r"\s+", " ", text).strip()


def load_descriptions(db_id: str) -> dict:
    """{(table, column) lower-cased: description row}"""
    out = {}
    for f in (DB_ROOT / db_id / "database_description").glob("*.csv"):
        table = f.stem.lower()
        for r in read_csv_any_encoding(f):
            col = r.get("original_column_name", "").strip().lower()
            if col:
                out[(table, col)] = r
    return out


def short(v) -> str:
    s = repr(v)
    return s if len(s) <= MAX_VALUE_CHARS else s[:MAX_VALUE_CHARS] + "...'"


def column_values(conn, table: str, col: str) -> str:
    rows = conn.execute(
        f'SELECT "{col}", COUNT(*) AS c FROM "{table}" WHERE "{col}" IS NOT NULL '
        f'GROUP BY "{col}" ORDER BY c DESC LIMIT {MAX_LIST_ALL + 1}'
    ).fetchall()
    if not rows:
        return "values: (all NULL)"
    if len(rows) <= MAX_LIST_ALL:
        return "all values: [" + ", ".join(short(r[0]) for r in rows) + "]"
    return "example: [" + ", ".join(short(r[0]) for r in rows[:N_EXAMPLES]) + "]"


def build(db_id: str) -> str:
    conn = sqlite3.connect(f"file:{db_path(db_id).as_posix()}?mode=ro", uri=True)
    desc = load_descriptions(db_id)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    parts = []
    for t in tables:
        cols = conn.execute(f'PRAGMA table_info("{t}")').fetchall()        # cid, name, type, notnull, default, pk
        fks = conn.execute(f'PRAGMA foreign_key_list("{t}")').fetchall()   # id, seq, table, from, to, ...
        lines = []   # (code, comment); the comma goes before "--", as in BIRD's official schema
        for _, name, ctype, *_rest in cols:
            d = desc.get((t.lower(), name.lower()), {})
            notes = []
            full = d.get("column_name", "")
            if full and full.lower() != name.lower():
                notes.append(f"full name: {clean(full)}")
            if d.get("column_description"):
                notes.append(clean(d["column_description"]))
            notes.append(column_values(conn, t, name))
            if d.get("value_description"):
                notes.append(f"value meaning: {clean(d['value_description'])}")
            lines.append((f"    `{name}` {ctype or 'TEXT'}", " -- " + "; ".join(notes)))
        pk = [c[1] for c in sorted(cols, key=lambda c: c[5]) if c[5] > 0]
        if pk:
            lines.append(("    PRIMARY KEY (" + ", ".join(f"`{c}`" for c in pk) + ")", ""))
        for fk in fks:
            lines.append((f"    FOREIGN KEY (`{fk[3]}`) REFERENCES `{fk[2]}` (`{fk[4]}`)", ""))
        body = "\n".join(code + ("," if i < len(lines) - 1 else "") + comment
                         for i, (code, comment) in enumerate(lines))
        parts.append(f"CREATE TABLE `{t}` (\n{body}\n);")
    conn.close()
    return "\n\n".join(parts) + "\n"


def main():
    RICH_SCHEMA_DIR.mkdir(exist_ok=True)
    for d in sorted(p.name for p in DB_ROOT.iterdir() if p.is_dir()):
        text = build(d)
        (RICH_SCHEMA_DIR / f"{d}.txt").write_text(text, encoding="utf-8")
        print(f"{d}: {len(text):,} chars")


if __name__ == "__main__":
    main()
