"""
Compares code with similar chunks found during the retrieval stage
"""

from langsmith import traceable

from codebloatguard.config import CONVENTIONS_MODEL
from codebloatguard.llm import DailyQuotaExhausted, generate_json

VERDICTS = ("FOLLOWS", "DEVIATES")

_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "notes": {"type": "string"},
    },
    "required": ["verdict", "notes"],
}

_PROMPT = """Below is a new Python function, followed by existing functions \
from the same repository that do the closest thing to it.

Those existing functions are the standard. A linter already checks the rules \
someone wrote down, so ignore anything a linter would catch. Look only for \
patterns that exist nowhere except in these neighbours:

- naming, for the function and its parameters. If the repo has one_third and \
one_fifth, a function returning a quarter should be one_fourth, not part_four.
- error handling. If every neighbour raises on bad input, returning None \
instead is a deviation, and the reverse is too.
- argument order and count, across functions that take the same things.
- return shape. If the neighbours return a tuple and this returns a dict, \
say so.

NEW FUNCTION ({new_label}):
{new_code}

EXISTING FUNCTIONS FROM THIS REPO:
{neighbours}

Answer DEVIATES only when the neighbours visibly agree on a pattern and the \
new function breaks it. Name the pattern, quote the neighbour that shows it, \
and say what the new function should have done instead.

If the neighbours disagree with each other there is no precedent to break, so \
answer FOLLOWS. Two examples are thin evidence. Do not invent rules from \
general Python style guides, and report nothing you cannot point at in the \
code above.
"""


@traceable(name="conventions")
def check_conventions(new_code: str, neighbours: list[dict], new_label: str = "") -> tuple[str, str]:
    """Compare one function against the repo code nearest to it.

    `neighbours` is the retrieval output: dicts with "code" and "name". Returns
    (verdict, notes), where verdict is one of VERDICTS or "ERROR". Never raises,
    for the same reason judge() never raises.
    """
    if not neighbours:
        return "FOLLOWS", "no neighbouring code to compare against"

    rendered = "\n\n".join(
        f"# {n.get('name', 'unnamed')}\n{n['code']}" for n in neighbours
    )
    prompt = _PROMPT.format(
        new_code=new_code,
        new_label=new_label or "candidate",
        neighbours=rendered,
    )
    try:
        data = generate_json(CONVENTIONS_MODEL, prompt, _SCHEMA)
        return data["verdict"], data["notes"]
    except DailyQuotaExhausted:
        raise
    except Exception as e:
        return "ERROR", f"conventions call failed: {type(e).__name__}: {e}"
