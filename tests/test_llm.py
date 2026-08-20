"""
Backoff tests. No network: the callable being retried is a counter.

The distinction under test is the whole point of the module. Gemini reports
the per-minute limit and the per-day limit with the same RESOURCE_EXHAUSTED
status, and only one of them is worth waiting for. Treating them alike is what
turned a PR with a dozen candidate pairs into eight minutes of sleeping to
rediscover, once per pair, that the day's quota was gone.
"""

import pytest

from codebloatguard.llm import DailyQuotaExhausted, MAX_TRIES, _wait_for, with_backoff

MINUTE_LIMIT = (
    "429 RESOURCE_EXHAUSTED. quota_metric: generate_requests, "
    "quotaId: GenerateRequestsPerMinutePerProjectPerModel, retryDelay: 21s"
)
DAILY_LIMIT = (
    "429 RESOURCE_EXHAUSTED. quota_metric: generate_requests, "
    "quotaId: GenerateRequestsPerDayPerProjectPerModel, retryDelay: 30s"
)


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    """Assert on the delays rather than serving them."""
    slept = []
    monkeypatch.setattr("codebloatguard.llm.time.sleep", slept.append)
    return slept


def failing(times: int, message: str, then=lambda: "ok"):
    """A callable that raises `times` times, then succeeds."""
    state = {"n": 0}

    def call():
        state["n"] += 1
        if state["n"] <= times:
            raise RuntimeError(message)
        return then()

    call.calls = lambda: state["n"]
    return call


class TestSuccess:
    def test_returns_the_value(self):
        assert with_backoff(lambda: "ok") == "ok"

    def test_does_not_retry_a_call_that_worked(self, no_sleeping):
        with_backoff(lambda: "ok")
        assert no_sleeping == []


class TestPerMinuteLimit:
    def test_is_retried_and_can_succeed(self):
        call = failing(2, MINUTE_LIMIT)
        assert with_backoff(call) == "ok"
        assert call.calls() == 3

    def test_honours_the_delay_gemini_asks_for(self, no_sleeping):
        with_backoff(failing(1, MINUTE_LIMIT))
        assert no_sleeping == [22.0]

    def test_gives_up_after_max_tries(self):
        call = failing(MAX_TRIES, MINUTE_LIMIT)
        with pytest.raises(RuntimeError):
            with_backoff(call)
        assert call.calls() == MAX_TRIES


class TestPerDayLimit:
    """Waiting cannot return quota that resets on Google's clock."""

    def test_is_not_retried(self):
        call = failing(1, DAILY_LIMIT)
        with pytest.raises(DailyQuotaExhausted):
            with_backoff(call)
        assert call.calls() == 1

    def test_never_sleeps(self, no_sleeping):
        with pytest.raises(DailyQuotaExhausted):
            with_backoff(failing(1, DAILY_LIMIT))
        assert no_sleeping == []

    def test_message_names_the_ways_out(self):
        with pytest.raises(DailyQuotaExhausted, match="no-judge"):
            with_backoff(failing(1, DAILY_LIMIT))

    def test_keeps_the_original_error_attached(self):
        with pytest.raises(DailyQuotaExhausted) as caught:
            with_backoff(failing(1, DAILY_LIMIT))
        assert "RESOURCE_EXHAUSTED" in str(caught.value.__cause__)


class TestOtherErrors:
    def test_are_raised_immediately(self):
        call = failing(1, "400 INVALID_ARGUMENT: schema rejected")
        with pytest.raises(RuntimeError, match="INVALID_ARGUMENT"):
            with_backoff(call)
        assert call.calls() == 1

    def test_a_malformed_prompt_does_not_spend_the_quota_three_times(self, no_sleeping):
        with pytest.raises(RuntimeError):
            with_backoff(failing(1, "400 INVALID_ARGUMENT"))
        assert no_sleeping == []


class TestWaitFor:
    def test_prefers_the_delay_in_the_error(self):
        assert _wait_for(MINUTE_LIMIT, attempt=0) == 22.0

    def test_falls_back_to_exponential_backoff(self):
        assert [_wait_for("429 RESOURCE_EXHAUSTED", a) for a in range(3)] == [5, 10, 20]


class TestStagesDoNotSwallowIt:
    """Every stage converts failures to ERROR so a dead call cannot read as a
    clean pass. A daily exhaustion has to be the exception: swallowing it
    spends one call per remaining pair to learn the same thing, and reports
    them all as ERROR as though they had been looked at."""

    @pytest.mark.parametrize(
        "module,func,args",
        [
            ("judge", "judge", ("def a(): pass", "def b(): pass")),
            ("conventions", "check_conventions",
             ("def a(): pass", [{"code": "def b(): pass", "name": "b"}])),
            ("triage", "triage", ("def a(): pass", "[0] b", 1, 0.30, 1, 3)),
        ],
    )
    def test_daily_quota_propagates(self, monkeypatch, module, func, args):
        import importlib

        mod = importlib.import_module(f"codebloatguard.{module}")
        monkeypatch.setattr(
            mod, "generate_json",
            lambda *a, **k: (_ for _ in ()).throw(DailyQuotaExhausted("gone")),
        )
        with pytest.raises(DailyQuotaExhausted):
            getattr(mod, func)(*args)

    @pytest.mark.parametrize(
        "module,func,args,expected",
        [
            ("judge", "judge", ("def a(): pass", "def b(): pass"), "ERROR"),
            ("conventions", "check_conventions",
             ("def a(): pass", [{"code": "def b(): pass", "name": "b"}]), "ERROR"),
            ("triage", "triage", ("def a(): pass", "[0] b", 1, 0.30, 1, 3), "ERROR"),
        ],
    )
    def test_ordinary_failure_still_becomes_error(self, monkeypatch, module, func, args, expected):
        import importlib

        mod = importlib.import_module(f"codebloatguard.{module}")
        monkeypatch.setattr(
            mod, "generate_json",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("timeout")),
        )
        assert getattr(mod, func)(*args)[0] == expected
