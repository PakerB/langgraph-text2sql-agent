"""LangGraph agent.

START -> prepare_prompt -> agent <-> tools -> check_final_sql -> (revise: agent) -> submit -> END

prepare_prompt, check_final_sql and submit are fixed workflow steps; the agent node decides
whether, what and how often to explore the database with run_sql.

v2 runs this graph once per reasoning strategy (STRATEGIES below) and picks one answer with
select_unit_tests.py. Without a strategy the graph is the v1 agent.
"""
import json
import re
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph

from .common import (
    FINAL_TIMEOUT_S, MAX_REVISIONS, MAX_ROWS, MAX_TOOL_CALLS, RICH_SCHEMA_DIR, TOOL_TIMEOUT_S,
    execute, extract_sql, fill_template, format_result, invoke_with_retry, message_text, official_question_block,
)

# Appended to the "Instructions:" section of the official BIRD prompt; the rest is unchanged.
EXTRA_INSTRUCTIONS = f"""
- The evidence (if any, the line before the question) explains meanings and formulas. It is not SQL syntax: rewrite it in valid SQLite (for example, SQLite has no DIVIDE function).
- You can call the tool `run_sql` to run read-only SQL on this database. Before filtering on a text value, check how it is really stored (exact spelling, case, format) with queries like SELECT DISTINCT col FROM t or WHERE col LIKE '%...%'. You may also test parts of your query. You have at most {MAX_TOOL_CALLS} tool calls.
- When you are done, reply with the final SQL in a ```sql code block and do not call any tool in that reply."""

# Optional checklist of generic SQL mistakes (no table, column or question specific content).
PITFALLS = """

Common SQL pitfalls to check before your final answer:
- Join fan-out: joining a one-to-many table repeats rows. When counting, summing or averaging over entities (people, items...), make sure each entity is counted once (COUNT(DISTINCT ...), or filter with IN (subquery) instead of JOIN).
- A text cell may hold several values (e.g. 'A, B'). If DISTINCT shows such lists, match the element, not the whole cell.
- NULLs: ORDER BY ... ASC puts NULLs first; MIN/MAX/AVG ignore them. Exclude NULLs when the question needs real values.
- "Latest/first value of a measure" means the latest record where that measure is not NULL, not simply the latest record.
- "For every X" must keep X with no matching rows (LEFT JOIN, COALESCE(..., 0)).
- Do not add filters the question does not ask for.
- If the evidence gives a formula, compute it as given instead of using a similar-looking stored column."""


# ---------- v2 reasoning strategies ----------
# What the model must write before the SQL. All wording is generic: no table, column or question content.
GRAIN_NOTES = """
- Start your first reply with a short "Notes:" section (at most 5 lines): what one row of the answer represents; what is counted, summed or averaged and at which grain (e.g. people, cards, single items, distinct orders); which columns hold it; and whether one stored row or cell is exactly one such unit (a cell may hold a list of several items, a table may repeat the same entity). Then use run_sql to check what you are unsure about."""

DECOMPOSE = """
- Start your first reply with a short "Sub-questions:" section: break the question into the smaller questions it depends on (e.g. which rows qualify, what is computed for them, how results are combined), each with a short SQL draft. Use run_sql to check the sub-queries you are unsure about, then combine them into one final query and check that its result agrees with the sub-results.
- Output columns: return exactly the values the question asks for, one column per requested value, in the order asked. Add no extra columns (no helper counts, ids, or values used only for sorting or filtering) and leave none out. If the question asks "how many", return the count, not the list."""

# Each strategy is exactly the configuration that was measured on the development set.
# `direct` has no list-column annotation: it was measured before the annotation existed and is kept as measured.
STRATEGIES = {
    "direct": {"pitfalls": True, "list_cols": False, "reasoning": ""},
    "grain": {"pitfalls": True, "list_cols": True, "reasoning": GRAIN_NOTES},
    "decompose": {"pitfalls": True, "list_cols": True, "reasoning": DECOMPOSE},
}


# ---------- List-column annotation ----------
LIST_COLUMNS_FILE = RICH_SCHEMA_DIR / "list_columns.json"   # written once by list_columns.py
_list_cols = None


def annotate_list_columns(schema: str, db_id: str) -> str:
    """Append a value-format note to the schema line of every column that list_columns.py classified as a list."""
    global _list_cols
    if _list_cols is None:
        if not LIST_COLUMNS_FILE.exists():
            raise SystemExit(f"{LIST_COLUMNS_FILE} not found. Run: python -m text2sql_agent.list_columns")
        _list_cols = json.loads(LIST_COLUMNS_FILE.read_text(encoding="utf-8"))
    tags = {k: v for k, v in _list_cols.get(db_id, {}).items() if v["is_list"]}
    if not tags:
        return schema
    out, table = [], None
    for line in schema.split("\n"):
        m = re.match(r"CREATE TABLE `(.+)` \(", line)
        if m:
            table = m.group(1)
        m = re.match(r"    `(.+?)` ", line)
        if m and f"{table}.{m.group(1)}" in tags:
            t = tags[f"{table}.{m.group(1)}"]
            line += (f"; value format: a cell may hold a list of several items separated by {t['separator']!r}, "
                     f"with or without a space after it (e.g. {t['example'][0]!r}); to count or match single items, "
                     f"split the cell and trim each item")
        out.append(line)
    return "\n".join(out)


def build_prompt(q: dict, pitfalls: bool = False, list_cols: bool = False, reasoning: str = "") -> str:
    """Official BIRD prompt with the enriched schema and the extra instructions."""
    schema = (RICH_SCHEMA_DIR / f"{q['db_id']}.txt").read_text(encoding="utf-8")
    if list_cols:
        schema = annotate_list_columns(schema, q["db_id"])
    prompt = fill_template(schema, official_question_block(q))
    extra = EXTRA_INSTRUCTIONS + reasoning + (PITFALLS if pitfalls else "")
    return prompt.replace("\n\nOutput Format:", extra + "\n\nOutput Format:", 1)


@tool
def run_sql(query: str) -> str:
    """Run a read-only SQLite query on the current database and return at most 20 rows.
    Use it to look at real stored values before filtering, or to test part of a query."""
    # Declaration only (name, argument, description for the model). Calls are executed in
    # the `tools` node, which knows which database the current question uses.
    raise RuntimeError("run_sql is executed by the graph's tools node")


class State(MessagesState):
    question: dict                 # one row of mini_dev_sqlite.jsonl (the gold SQL field is never used)
    tool_calls_used: int
    revisions: int
    last_checked_sql: Optional[str]
    final_sql: Optional[str]


def make_graph(llm, pitfalls: bool = False, strategy: Optional[str] = None):
    """v1 agent (optionally with the pitfalls checklist), or one v2 strategy (which sets everything itself)."""
    cfg = STRATEGIES[strategy] if strategy else {"pitfalls": pitfalls, "list_cols": False, "reasoning": ""}
    llm_with_tools = llm.bind_tools([run_sql])

    def prepare_prompt(state: State):
        return {"messages": [HumanMessage(build_prompt(state["question"], **cfg))],
                "tool_calls_used": 0, "revisions": 0, "last_checked_sql": None, "final_sql": None}

    def agent(state: State):
        if state["tool_calls_used"] >= MAX_TOOL_CALLS:
            # Tool budget exhausted: call the model without tools to force a final answer.
            note = HumanMessage("You have used all tool calls. Write the final SQL now in a ```sql code block.")
            return {"messages": [note, invoke_with_retry(llm, state["messages"] + [note])]}
        return {"messages": [invoke_with_retry(llm_with_tools, state["messages"])]}

    def route_after_agent(state: State):
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else "check_final_sql"

    def tools(state: State):
        db_id = state["question"]["db_id"]
        out = []
        for tc in state["messages"][-1].tool_calls:
            args = tc.get("args") or {}
            if tc["name"] != "run_sql":
                text = f"ERROR: unknown tool {tc['name']}"
            elif isinstance(args.get("query"), str) and args["query"].strip():
                text = format_result(*execute(db_id, args["query"], TOOL_TIMEOUT_S, max_rows=MAX_ROWS))
            elif isinstance(args.get("queries"), list) and args["queries"]:
                # The model sometimes sends a list of queries; run up to 3 and restate the signature.
                parts = [f"Query {k}: {s}\n" + format_result(*execute(db_id, str(s), TOOL_TIMEOUT_S, max_rows=MAX_ROWS))
                         for k, s in enumerate(args["queries"][:3], 1)]
                text = "\n\n".join(parts) + "\n(Note: run_sql takes one argument `query`, a single SQL string.)"
            else:
                text = "ERROR: run_sql takes one argument `query` (a single SQL string)."
            out.append(ToolMessage(content=text, tool_call_id=tc["id"]))
        return {"messages": out, "tool_calls_used": state["tool_calls_used"] + len(out)}

    def check_final_sql(state: State):
        """Execute the final SQL; ask for a revision on a missing block, an error, or a suspicious result."""
        sql = extract_sql(message_text(state["messages"][-1].content))
        problem = None
        if sql is None:
            problem = "I could not find a ```sql code block in your reply. Reply with the final SQL in a ```sql code block."
        else:
            _, rows, _, error = execute(state["question"]["db_id"], sql, FINAL_TIMEOUT_S, max_rows=1)
            if error:
                problem = f"Your final SQL failed with: {error}\nFix it and reply with the corrected SQL in a ```sql code block."
            elif sql != state["last_checked_sql"] and (not rows or all(v is None for v in rows[0])):
                # Empty or all-NULL results (e.g. AVG over no rows) usually mean a wrong filter or join,
                # but can be correct: resubmitting the same SQL is accepted.
                what = "0 rows" if not rows else "a first row where every value is NULL"
                problem = (f"Your final SQL returned {what}. This often means a filter or join condition is wrong. "
                           "Check the filter values and join conditions against the real data (you may use run_sql). "
                           "If you are sure this result is correct, reply with the same SQL again.")
        best = sql or state["final_sql"]
        if problem and state["revisions"] < MAX_REVISIONS:
            return {"messages": [HumanMessage(problem)], "revisions": state["revisions"] + 1,
                    "last_checked_sql": sql, "final_sql": best}
        return {"final_sql": best, "last_checked_sql": sql}

    def route_after_check(state: State):
        return "agent" if isinstance(state["messages"][-1], HumanMessage) else "submit"

    def submit(state: State):
        return {}   # the answer is state["final_sql"]

    g = StateGraph(State)
    g.add_node("prepare_prompt", prepare_prompt)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_node("check_final_sql", check_final_sql)
    g.add_node("submit", submit)
    g.add_edge(START, "prepare_prompt")
    g.add_edge("prepare_prompt", "agent")
    g.add_conditional_edges("agent", route_after_agent, ["tools", "check_final_sql"])
    g.add_edge("tools", "agent")
    g.add_conditional_edges("check_final_sql", route_after_check, ["agent", "submit"])
    g.add_edge("submit", END)
    return g.compile()


def solve(graph, q: dict) -> dict:
    state = graph.invoke({"question": q, "messages": []}, config={"recursion_limit": 60})
    return {"sql": state["final_sql"], "messages": state["messages"],
            "tool_calls_used": state["tool_calls_used"], "revisions": state["revisions"]}
