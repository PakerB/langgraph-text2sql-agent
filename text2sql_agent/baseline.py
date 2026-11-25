"""Official BIRD baseline: fixed prompt and official schema, one model call, no tools."""
import json

from langchain_core.messages import HumanMessage

from .common import (OFFICIAL_SCHEMAS_FILE, extract_sql, fill_template, invoke_with_retry, message_text,
                     official_question_block)

_SCHEMAS = None


def build_prompt(q: dict) -> str:
    global _SCHEMAS
    if _SCHEMAS is None:
        _SCHEMAS = json.loads(OFFICIAL_SCHEMAS_FILE.read_text(encoding="utf-8"))
    return fill_template(_SCHEMAS[q["db_id"]], official_question_block(q))


def solve(llm, q: dict) -> dict:
    prompt = HumanMessage(build_prompt(q))
    msg = invoke_with_retry(llm, [prompt])
    return {"sql": extract_sql(message_text(msg.content)), "messages": [prompt, msg]}
