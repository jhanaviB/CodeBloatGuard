from langsmith import traceable

from codebloatguard.config import JUDGE_MODEL
from codebloatguard.llm import DailyQuotaExhausted, generate_json

VERDICTS = ("DUPLICATE", "SIMILAR", "DIFFERENT")
ACTIONS = ("REPLACE", "EXTRACT", "KEEP_BOTH", "NONE")

_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "reason": {"type": "string"},
        "action": {"type": "string", "enum": list(ACTIONS)},
        "advice": {"type": "string"},
    },
    "required": ["verdict", "reason", "action", "advice"],
}

_PROMPT = """You are reviewing a Python codebase for duplicated logic.

NEW CODE ({new_label}):
{new_code}

EXISTING CODE ({old_label}):
{old_code}

First classify the relationship:
- DUPLICATE: same logic and same result. Differences are only naming, style, \
formatting, or a trivially equivalent expression (x/2 vs x*0.5). Calling the \
existing one instead would lose nothing.
- SIMILAR: real overlap worth a look (one wraps or nearly contains the other, \
or they share a substantial block, but they are not interchangeable).
- DIFFERENT: they merely resemble each other. Same shape, different purpose. \
Inverses (is_even vs is_odd) are DIFFERENT.

Judge behaviour, not vocabulary. Give one sentence naming the specific evidence.

Then pick the action:
- REPLACE: the new function adds nothing. Delete it and call the existing one.
- EXTRACT: both are needed but share a block worth pulling into one helper. \
Say what the helper should contain.
- KEEP_BOTH: the overlap is real but separating it would cost more than it \
saves, for example when the shared part is two lines.
- NONE: use this whenever the verdict is DIFFERENT. Leave advice empty.

Do not let the existence of an action list push you toward a stronger verdict. \
If these two functions do different work, the answer is DIFFERENT and NONE. \
Otherwise give one sentence a reviewer could act on without reading either \
function again, naming the function that survives.
"""


@traceable(name="judge")
def judge(
    new_code: str, old_code: str, new_label: str = "", old_label: str = ""
) -> tuple[str, str, str, str]:
    """Classify one pair and say what to do about it.
    VERDICTS = ("DUPLICATE", "SIMILAR", "DIFFERENT")
    ACTIONS = ("REPLACE", "EXTRACT", "KEEP_BOTH", "NONE")
    API errors or other failures with model come back as "ERROR"
    """
    prompt = _PROMPT.format(
        new_code=new_code,
        old_code=old_code,
        new_label=new_label or "candidate",
        old_label=old_label or "indexed",
    )
    try:
        data = generate_json(JUDGE_MODEL, prompt, _SCHEMA)
        return data["verdict"], data["reason"], data["action"], data["advice"]
    except DailyQuotaExhausted:
        # Not this pair's failure, and not survivable by trying the next
        # one. Swallowing it would spend a call per remaining pair to
        # rediscover it, and report them all as ERROR as if they differed.
        raise
    except Exception as e:
        return "ERROR", f"judge call failed: {type(e).__name__}: {e}", "NONE", ""
