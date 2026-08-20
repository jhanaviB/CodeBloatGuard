"""
Claude provider parsing and validation. No session, no network.

The Gemini path gets a schema honoured server side; the Claude path asks for
one in a system prompt and hopes. So everything the Gemini path takes for
granted has to be checked on arrival here, and these are the checks.

The failure that matters is not a crash. Every stage catches broadly and
converts to an ERROR verdict, so a reply of {"verdict": "MAYBE"} that slipped
through would reach code branching on three known strings, match none of them,
and be reported as a clean pass. A rate-limited key must never look like
reviewed-and-fine, and neither must a creative one.
"""

import pytest

from codebloatguard.claude import _check, _extract_json

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["DUPLICATE", "SIMILAR", "DIFFERENT"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
}


class TestExtractJson:
    def test_bare_object(self):
        assert _extract_json('{"verdict": "DUPLICATE"}') == {"verdict": "DUPLICATE"}

    def test_tolerates_surrounding_whitespace(self):
        assert _extract_json('\n  {"verdict": "SIMILAR"}\n ') == {"verdict": "SIMILAR"}

    def test_tolerates_a_markdown_fence(self):
        """Asking for bare JSON usually works. A fence is the common enough
        miss that retrying the call costs more than allowing for it."""
        reply = '```json\n{"verdict": "DIFFERENT", "reason": "inverse"}\n```'
        assert _extract_json(reply)["verdict"] == "DIFFERENT"

    def test_tolerates_a_sentence_of_preamble(self):
        reply = 'Here is my assessment:\n{"verdict": "DUPLICATE", "reason": "same"}'
        assert _extract_json(reply)["verdict"] == "DUPLICATE"

    def test_keeps_nested_objects_intact(self):
        reply = '{"verdict": "SIMILAR", "meta": {"confidence": {"score": 1}}}'
        assert _extract_json(reply)["meta"]["confidence"]["score"] == 1

    def test_reply_with_no_object_raises(self):
        with pytest.raises(ValueError, match="no JSON object"):
            _extract_json("I am unable to classify these two functions.")

    def test_empty_reply_raises(self):
        with pytest.raises(ValueError):
            _extract_json("")


class TestSchemaCheck:
    def test_valid_payload_passes_through(self):
        data = {"verdict": "DUPLICATE", "reason": "same arithmetic"}
        assert _check(data, SCHEMA) == data

    def test_missing_required_key_raises(self):
        """judge() unpacks four keys positionally. A short dict is a KeyError
        somewhere less obvious than here."""
        with pytest.raises(ValueError, match="missing key 'reason'"):
            _check({"verdict": "DUPLICATE"}, SCHEMA)

    def test_invented_enum_value_raises(self):
        """The one that would otherwise read as a clean pass: no branch
        matches 'MAYBE', so nothing is reported and nothing looks wrong."""
        with pytest.raises(ValueError, match="not one of"):
            _check({"verdict": "MAYBE", "reason": "unsure"}, SCHEMA)

    def test_enum_matching_is_case_sensitive(self):
        with pytest.raises(ValueError, match="not one of"):
            _check({"verdict": "duplicate", "reason": "same"}, SCHEMA)

    def test_extra_keys_are_allowed(self):
        """Callers read the keys they asked for. An extra one is harmless and
        rejecting it would fail a verdict that is otherwise usable."""
        data = {"verdict": "SIMILAR", "reason": "overlap", "confidence": 0.9}
        assert _check(data, SCHEMA)["verdict"] == "SIMILAR"

    def test_unconstrained_field_accepts_any_string(self):
        assert _check({"verdict": "SIMILAR", "reason": ""}, SCHEMA)["reason"] == ""


class TestJudgeSchemaIsEnforceable:
    """The real judge schema, checked against the guard that protects it."""

    def test_every_verdict_and_action_is_accepted(self):
        from codebloatguard.judge import ACTIONS, VERDICTS, _SCHEMA

        for verdict in VERDICTS:
            for action in ACTIONS:
                payload = {
                    "verdict": verdict,
                    "reason": "r",
                    "action": action,
                    "advice": "a",
                }
                assert _check(payload, _SCHEMA) == payload

    def test_a_plausible_invention_is_rejected(self):
        from codebloatguard.judge import _SCHEMA

        with pytest.raises(ValueError):
            _check(
                {"verdict": "NEARLY", "reason": "r", "action": "REPLACE", "advice": "a"},
                _SCHEMA,
            )
