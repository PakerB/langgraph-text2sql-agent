"""Shared configuration and helpers: paths, limits, safe SQL execution, SQL extraction, model setup."""
import json
import os
import re
import sqlite3
import time
from pathlib import Path

# ---------- Paths ----------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("BIRD_DATA_DIR", REPO_ROOT / "data"))
QUESTIONS_FILE = DATA_DIR / "mini_dev_sqlite.jsonl"
OFFICIAL_SCHEMAS_FILE = DATA_DIR / "schemas_sqlite.json"
PROMPT_TEMPLATE_FILE = DATA_DIR / "prompt_template.txt"
DB_ROOT = Path(os.environ.get("BIRD_DB_ROOT", DATA_DIR / "dev_databases"))
RICH_SCHEMA_DIR = REPO_ROOT / "schemas"
RUNS_DIR = REPO_ROOT / "runs"

# ---------- Limits ----------
MAX_ROWS = 20              # rows returned by run_sql
MAX_CELL_CHARS = 100       # characters per cell returned by run_sql
TOOL_TIMEOUT_S = 10        # exploration queries
FINAL_TIMEOUT_S = 30       # final SQL check and evaluation (slowest gold query takes ~12 s)
MAX_TOOL_CALLS = 6         # tool calls per question
MAX_REVISIONS = 3          # revision rounds after the final-SQL check

DEFAULT_MODEL = "gemini-3.5-flash-lite"


def db_path(db_id: str) -> Path:
    return DB_ROOT / db_id / f"{db_id}.sqlite"


def load_questions() -> list[dict]:
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_env_key(name: str = "GEMINI_API_KEY") -> str:
    """Read the API key from the environment or from ./.env (never printed)."""
    if os.environ.get(name):
        return os.environ[name]
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(name + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(f"{name} is not set. Add it to {env_file} (see .env.example).")


# ---------- Safe SQL execution ----------
def execute(db_id: str, sql: str, timeout_s: float, max_rows: int | None):
    """Run read-only SQL with a timeout.

    Returns (columns, rows, has_more, error); error is None on success.
    max_rows=None fetches everything (used for scoring).
    """
    if not (sql or "").strip():
        return [], [], False, "empty SQL"
    conn = sqlite3.connect(f"file:{db_path(db_id).as_posix()}?mode=ro", uri=True)
    deadline = time.monotonic() + timeout_s
    # SQLite calls this periodically; returning 1 interrupts the query.
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        if max_rows is None:
            return cols, cur.fetchall(), False, None
        rows = cur.fetchmany(max_rows)
        has_more = cur.fetchone() is not None
        return cols, rows, has_more, None
    except sqlite3.OperationalError as e:
        msg = str(e)
        if msg == "interrupted":
            msg = f"query timed out after {timeout_s:.0f} seconds"
        return [], [], False, msg
    except Exception as e:  # noqa: BLE001 - every error is reported back to the model as text
        return [], [], False, f"{type(e).__name__}: {e}"
    finally:
        conn.close()


def _cell(v) -> str:
    s = repr(v)
    return s if len(s) <= MAX_CELL_CHARS else s[:MAX_CELL_CHARS] + "...'"


def format_result(cols, rows, has_more, error) -> str:
    """Render a query result as text for the model."""
    if error:
        return f"ERROR: {error}"
    if not rows:
        return f"columns: {cols}\n(0 rows)"
    lines = [f"columns: {cols}"] + [" | ".join(_cell(v) for v in r) for r in rows]
    if has_more:
        lines.append(f"(more rows exist; only the first {MAX_ROWS} are shown)")
    return "\n".join(lines)


# ---------- SQL extraction ----------
_SQL_OPEN = re.compile(r"```sql", re.IGNORECASE)


def message_text(content) -> str:
    """Message content may be a string or a list of parts; join the text parts."""
    if isinstance(content, str):
        return content
    parts = []
    for p in content or []:
        if isinstance(p, dict) and p.get("type") == "text":
            parts.append(p.get("text", ""))
        elif isinstance(p, str):
            parts.append(p)
    return "\n".join(parts)


def extract_sql(text: str) -> str | None:
    """Return the last non-empty ```sql block.

    Each ```sql opening is read up to the next ``` fence, so a stray duplicated
    opening line yields an empty block that is skipped.
    """
    text = text or ""
    blocks = []
    for m in _SQL_OPEN.finditer(text):
        end = text.find("```", m.end())
        code = text[m.end(): end if end != -1 else len(text)].strip()
        if code:
            blocks.append(code)
    return blocks[-1] if blocks else None


# ---------- Model ----------
def is_busy_error(e: Exception) -> bool:
    """Transient server-side errors (overload, timeouts, network)."""
    text = f"{type(e).__name__} {e}"
    return any(s in text for s in ("503", "500 ", "UNAVAILABLE", "INTERNAL", "Timeout", "timed out", "Connection"))


def invoke_with_retry(llm, messages, tries: int = 6, wait_s: float = 20):
    """Retry only the failed model call on transient errors; re-raise anything else (e.g. 429)."""
    for i in range(tries):
        try:
            return llm.invoke(messages)
        except Exception as e:  # noqa: BLE001
            if not is_busy_error(e) or i == tries - 1:
                raise
            print(f"    (server busy, retrying this call in {wait_s * (i + 1):.0f} s)")
            time.sleep(wait_s * (i + 1))


DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def make_llm(model: str, thinking: bool = False):
    """Create the chat model from its name.

    - "deepseek-*": OpenAI-compatible API, non-thinking mode by default. Thinking mode cannot be used
      with tool calls here: the API then requires the previous turns' reasoning_content to be sent back,
      which langchain-openai drops. thinking=True is used only for single calls without tools
      (the thinking baseline and the unit-test selector). No per-minute limit, so no rate limiter.
    - otherwise Gemini via langchain-google-genai. The native client is required for tool calling:
      Gemini returns a thought signature with each function call that must be sent back on the next turn.
    """
    if model.startswith("deepseek"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=load_env_key("DEEPSEEK_API_KEY"),
            base_url=DEEPSEEK_BASE_URL,
            temperature=0,
            max_retries=2,
            timeout=300 if thinking else 120,
            extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        )

    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_google_genai import ChatGoogleGenerativeAI

    # Free tier allows 15 requests/minute; stay slightly below.
    limiter = InMemoryRateLimiter(requests_per_second=14 / 60, check_every_n_seconds=0.2, max_bucket_size=1)
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=load_env_key(),
        temperature=0,
        max_retries=2,
        rate_limiter=limiter,
        timeout=120,
    )


# ---------- Official BIRD prompt ----------
def official_question_block(q: dict) -> str:
    """Evidence, a newline, then the question (question only when there is no evidence)."""
    return (q["evidence"] + "\n" if q.get("evidence") else "") + q["question"]


def fill_template(schema: str, question_block: str) -> str:
    t = PROMPT_TEMPLATE_FILE.read_text(encoding="utf-8")
    # str.replace instead of str.format: schemas and questions may contain braces.
    return t.replace("{engine}", "SQLite").replace("{schema}", schema).replace("{question_block}", question_block)
