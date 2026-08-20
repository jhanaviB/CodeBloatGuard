"""
LLM call made by judge, triage and conventions goes through here.
"""

import json
import os
import re
import time

from google import genai
from langsmith import traceable

from codebloatguard.config import JUDGE_PROVIDER

_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MAX_TRIES = 3
_RETRY_AFTER = re.compile(r"retryDelay['\"]?:\s*['\"]?(\d+)")

# Gemini reports both the per-minute and the per-day limit as
# RESOURCE_EXHAUSTED, and names which one in the quota id. Only one of them is
# worth waiting for.
_DAILY_QUOTA = re.compile(r"PerDay", re.IGNORECASE)


class DailyQuotaExhausted(RuntimeError):
    """The daily generation limit is gone. Nothing recovers this today."""


def _wait_for(error: str, attempt: int) -> float:
    """Seconds to wait. Prefers the delay Gemini asked for, falls back to
    exponential backoff when the error does not carry one."""
    m = _RETRY_AFTER.search(error)
    return float(m.group(1)) + 1 if m else 2 ** attempt * 5


def with_backoff(call):
    """Run a zero-argument callable, retrying rate limits only.

    Shared by generation and embedding, because both hit the same free tier
    minute limits and a run that dies on the first 429 cannot index anything
    larger than a toy repo.

    A per-day exhaustion is not retried. Sleeping cannot return quota that
    resets tomorrow, so every call after the first would spend MAX_TRIES
    attempts and the delay Gemini asks for to rediscover the same fact. On a
    PR with a dozen candidate pairs that turned a run into eight minutes of
    sleeping, which is how this was found.
    """
    for attempt in range(MAX_TRIES):
        try:
            return call()
        except Exception as e:
            text = str(e)
            if "RESOURCE_EXHAUSTED" not in text:
                raise
            if _DAILY_QUOTA.search(text):
                raise DailyQuotaExhausted(
                    "daily generation quota exhausted; it resets on Google's "
                    "clock, not by waiting. Re-run tomorrow, use --no-judge "
                    "for retrieval distances at no cost, or set "
                    "CBG_PROVIDER=claude."
                ) from e
            if attempt == MAX_TRIES - 1:
                raise
            time.sleep(_wait_for(text, attempt))


def generate_json(model: str, prompt: str, schema: dict) -> dict:
    """Route one structured call to the configured provider.

    The signature is the seam every stage is written against, so switching
    providers is a config change and touches no prompt, schema or caller.
    """
    if JUDGE_PROVIDER == "claude":
        from codebloatguard import claude

        return claude.generate_json(model, prompt, schema)

    elif JUDGE_PROVIDER == "openrouter":
        from codebloatguard import openrouter
        return openrouter.generate_json(model, prompt, schema)

    elif JUDGE_PROVIDER != "gemini":
        raise ValueError(
            f"unknown CBG_PROVIDER {JUDGE_PROVIDER!r}, expected 'gemini' or 'claude'"
        )

    return _generate_json_model(model, prompt, schema)


@traceable(name="model", run_type="llm")
def _generate_json_model(model: str, prompt: str, schema: dict) -> dict:
    """Call the model and parse its JSON response.

    Traced so every prompt, response and retry is inspectable in LangSmith.
    Tracing is a no-op unless LANGSMITH_TRACING=true is set, so this costs
    nothing when observability is off.

    Retries on rate limiting only. Any other failure is raised immediately,
    because retrying a malformed prompt just spends the quota that the next
    real call needs. Callers catch and convert to their own ERROR verdict.
    """
    def call():
        r = _client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": schema,
                "temperature": 0,
            },
        )
        return json.loads(r.text)

    return with_backoff(call)
