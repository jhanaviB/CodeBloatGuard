"""
For local runs: Claude judge, backed by the Agent SDK rather than the Anthropic API.

The Agent SDK resolves credentials the same way the Claude Code CLI does, so
this runs on a local Claude Pro login and spends no API credits. 
"""

import asyncio
import json
import re

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

_INSTRUCTION = (
    "You are a code review classifier. Reply with a single JSON object and "
    "nothing else: no prose, no explanation, no markdown fence.\n\n"
    "The object must use exactly these keys, spelled exactly this way: "
    "{required}. Do not rename them, do not substitute synonyms, and do not "
    "omit any of them. Extra keys are ignored. Any field with an enum must "
    "hold one of the listed values verbatim.\n\n"
    "Schema:\n{schema}"
)


async def _ask(system_prompt: str, prompt: str) -> str:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        TextBlock,
        query,
    )

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        # No tools. This is a classifier, and every tool definition is more of
        # the prompt to cache and more for the model to weigh.
        allowed_tools=[],
        # Skip CLAUDE.md and project settings. A verdict must not depend on
        # whose checkout it ran in.
        setting_sources=[],
        # Not 1. With no tools there is no loop to bound, so this only decides
        # whether the model may think before answering, and at 1 a stage that
        # reasons before it commits gets cut off mid-turn and returns an error
        # instead of a verdict.
        max_turns=3,
    )

    text = ""
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    text += block.text
    return text


def _extract_json(text: str) -> dict:
    """Pull the JSON object out of a reply.

    Asking for bare JSON usually gets bare JSON, but a fenced block or a
    sentence of preamble is a normal enough failure that retrying the call
    costs more than tolerating it here.
    """
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    match = _JSON_BLOCK.search(text)
    if not match:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    return json.loads(match.group(0))


def _check(data: dict, schema: dict) -> dict:
    """Enforce the parts of the schema the caller relies on.

    The Gemini path gets this from response_schema server side. Here the model
    is only asked politely, so the same guarantees are checked on arrival:
    every required key present, every enum value one the caller listed.
    Downstream code branches on these strings, and a plausible-looking
    invention like "MAYBE" would fall through every branch silently.
    """
    for key in schema.get("required", []):
        if key not in data:
            raise ValueError(f"missing key {key!r} in {sorted(data)}")

    for key, spec in schema.get("properties", {}).items():
        if "enum" in spec and key in data and data[key] not in spec["enum"]:
            raise ValueError(f"{key}={data[key]!r} not one of {spec['enum']}")

    return data


def generate_json(model: str, prompt: str, schema: dict) -> dict:
    """
    `model` is ignored. The Agent SDK uses whatever model the
    local Claude Code install is configured with.
    """
    required = ", ".join(repr(k) for k in schema.get("required", [])) or "as in the schema"
    instruction = _INSTRUCTION.format(
        required=required, schema=json.dumps(schema, indent=2)
    )

    text = asyncio.run(_ask(instruction, f"{prompt}\n\n---\n\n{instruction}"))
    return _check(_extract_json(text), schema)
