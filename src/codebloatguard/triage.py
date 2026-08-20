from langsmith import traceable

from codebloatguard.config import TRIAGE_MODEL
from codebloatguard.llm import DailyQuotaExhausted, generate_json

ACTIONS = ("WIDEN", "JUDGE", "STOP") 

_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(ACTIONS)},
        "max_distance": {"type": "number"},
        "keep": {"type": "array", "items": {"type": "integer"}},
        "reason": {"type": "string"},
    },
    "required": ["action", "keep", "reason"],
}

_PROMPT = """You are deciding how much effort to spend checking whether one \
Python function duplicates something already in this repository.

Retrieval returned the {n} nearest existing functions by embedding distance. \
Distance is 1 minus cosine similarity, so 0 means identical and larger means \
less alike. Anything under {max_distance:.2f} is considered close enough to be \
worth judging by default, but you are shown the rest as well.

This is attempt {attempt} of {max_attempts}. Choosing WIDEN retrieves a larger \
number of neighbours, so it is worth doing only when the furthest candidate \
here is still plausibly related, which suggests the list was cut off before \
the real match.

FUNCTION UNDER REVIEW:
{code}

NEAREST EXISTING FUNCTIONS:
{candidates}

Choose one action:
- JUDGE: at least one candidate plausibly has overlapping code/does the same work as the function \
under review. Put those candidate numbers in "keep". Include anything worth a \
careful read, including candidates past the current radius. Judging is the \
expensive step, so do not select candidates that merely share a topic.
- WIDEN: nothing retrieved is close enough to judge, but the furthest \
candidate is still plausibly related, so the real match may lie just past the \
end of this list. Leave "keep" empty.
- STOP: the candidates are unrelated work that happens to look alike, or the \
distances have flattened out into noise. Leave "keep" empty.

Prefer STOP over WIDEN when the nearest candidate is already unrelated, since \
widening from there only retrieves things that are further away.

Prefer JUDGE over STOP when you are unsure. A pair sent to the judge \
unnecessarily costs one call and is then dismissed. A pair withheld from the \
judge is never looked at again by anything.

Give one sentence of reasoning naming the specific candidate that drove the \
decision.
"""


@traceable(name="triage")
def triage(
    code: str,
    candidates: str,
    n_candidates: int,
    max_distance: float,
    attempt: int,
    max_attempts: int,
) -> tuple[str, float, list[int], str]:
    """Decide what to do with one function's retrieval results.

    Returns (action, suggested_distance, keep_indices, reason). 
    On failure the action returned is ERROR, so the caller knows that the function needs to be re-checked
    """
    prompt = _PROMPT.format(
        code=code,
        candidates=candidates,
        n=n_candidates,
        max_distance=max_distance,
        attempt=attempt,
        max_attempts=max_attempts,
    )
    try:
        data = generate_json(TRIAGE_MODEL, prompt, _SCHEMA)
        return (
            data["action"],
            float(data.get("max_distance") or max_distance),
            list(data.get("keep") or []),
            data["reason"],
        )
    except DailyQuotaExhausted:
        raise
    except Exception as e:
        return "ERROR", max_distance, [], f"triage call failed: {type(e).__name__}: {e}"
