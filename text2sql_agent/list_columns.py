"""Find columns whose cells hold a list of several items (run once, after build_schema).

1. Code finds candidate columns: text columns where at least 5% of the distinct values contain a
   separator (',', ';' or '|').
2. One model call per database decides, for each candidate, whether a cell is a LIST of several items
   (e.g. several products bought) or a SINGLE value that happens to contain the separator
   (e.g. 'Corvallis, OR' is one address). Code alone cannot tell these apart.
3. The result goes to schemas/list_columns.json; the agent's `grain` and `decompose` strategies
   annotate these columns in the schema shown to the model.

Uses only the databases and the column descriptions, never questions or answers.
Usage:  python -m text2sql_agent.list_columns [--model gemini-3.5-flash-lite]
"""
import argparse
import json
import re
import sqlite3
import sys

from langchain_core.messages import HumanMessage

from .build_schema import clean, load_descriptions
from .common import DB_ROOT, DEFAULT_MODEL, RICH_SCHEMA_DIR, db_path, invoke_with_retry, make_llm, message_text

SEPARATORS = (",", ";", "|")
MIN_SHARE = 0.05        # share of distinct values containing the separator
MAX_DISTINCT = 5000
OUT_FILE = RICH_SCHEMA_DIR / "list_columns.json"


def candidates(db_id: str) -> list[dict]:
    conn = sqlite3.connect(f"file:{db_path(db_id).as_posix()}?mode=ro", uri=True)
    desc = load_descriptions(db_id)
    out = []
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    for t in tables:
        for _, col, *_ in conn.execute(f'PRAGMA table_info("{t}")').fetchall():
            vals = [r[0] for r in conn.execute(
                f'SELECT DISTINCT "{col}" FROM "{t}" WHERE typeof("{col}") = \'text\' LIMIT {MAX_DISTINCT}')]
            if len(vals) < 3:
                continue
            for sep in SEPARATORS:
                with_sep = [v for v in vals if sep in v]
                if len(with_sep) / len(vals) >= MIN_SHARE:
                    without = [v for v in vals if sep not in v]
                    d = desc.get((t.lower(), col.lower()), {})
                    out.append({
                        "table": t, "column": col, "separator": sep,
                        "share": round(len(with_sep) / len(vals), 2), "n_distinct": len(vals),
                        "description": clean(d.get("column_description", "")),
                        "with_sep": sorted(with_sep, key=len)[:12], "without_sep": without[:5],
                    })
                    break
    conn.close()
    return out


PROMPT = """Below are columns of a SQLite database whose text values often contain the separator shown.
For each column decide whether a cell is a LIST of several separate items of the same kind (for example several products bought, several diagnoses, several platforms), so that counting or matching single items requires splitting the cell; or a SINGLE value that just happens to contain the separator (a title, a sentence, free text, code, JSON). A value made of the parts of ONE thing is a SINGLE value: an address or place like 'City, State' or 'City, Country', a name like 'Last, First', a name of one department or organisation.

{columns}

Answer with JSON only, one entry per column, in this form:
{{"<table>.<column>": {{"is_list": true or false, "reason": "<at most 12 words>"}}}}"""


def short(v, n=80):
    s = repr(v)
    return s if len(s) <= n else s[:n] + "...'"


def classify(llm, cands: list[dict]) -> dict:
    blocks = []
    for c in cands:
        blocks.append(
            f"- {c['table']}.{c['column']} (separator {c['separator']!r}, {int(100 * c['share'])}% of {c['n_distinct']} distinct values contain it)\n"
            f"  description: {c['description'] or '(none)'}\n"
            f"  values with separator: {', '.join(short(v) for v in c['with_sep'])}\n"
            f"  values without: {', '.join(short(v) for v in c['without_sep']) or '(none)'}")
    for attempt in range(3):   # the model occasionally returns broken JSON: ask again
        reply = invoke_with_retry(llm, [HumanMessage(PROMPT.format(columns="\n".join(blocks)))])
        m = re.search(r"\{.*\}", message_text(reply.content), re.S)
        try:
            return json.loads(m.group(0))
        except (AttributeError, json.JSONDecodeError):
            print(f"  (invalid JSON, asking again: attempt {attempt + 2})")
    raise RuntimeError("no valid JSON after 3 attempts")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    llm = make_llm(ap.parse_args().model)
    result = {}
    for db in sorted(p.name for p in DB_ROOT.iterdir() if p.is_dir()):
        cands = candidates(db)
        if not cands:
            continue
        verdict = classify(llm, cands)
        for c in cands:
            key = f"{c['table']}.{c['column']}"
            v = verdict.get(key, {})
            result.setdefault(db, {})[key] = {"is_list": bool(v.get("is_list")), "separator": c["separator"],
                                              "reason": v.get("reason", ""), "example": c["with_sep"][:3]}
            print(f"{'LIST ' if v.get('is_list') else '     '}{db}.{key}  ({v.get('reason', '?')})")
    OUT_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
