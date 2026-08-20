"""
Graph tests. All three agents are stubbed, so this checks wiring rather than
verdicts: who runs, in what order, on which input, and what survives to the
end.

That split is the point. Whether the judge is right about two functions is
what `cbg eval` measures against labeled pairs, and it costs quota. Whether
conventions runs on a function that is about to be deleted is a wiring
question, it is free to answer, and it is the kind that breaks silently during
a refactor.
"""

import pytest

from codebloatguard import agent as agent_module


class FakeStore:
    """Returns fixed hits in store.search_vec's shape, and records the calls
    so widening can be observed."""

    def __init__(self, hits=None):
        self.hits = hits if hits is not None else []
        self.calls = []

    def search_vec(self, vector, k=5, exclude_paths=None):
        self.calls.append({"k": k, "exclude_paths": exclude_paths})
        return self.hits[:k]


def hit(name, code, distance, path="old.py"):
    return {
        "id": f"{path}:{name}:abc123",
        "code": code,
        "meta": {"path": path, "name": name, "start_line": 1, "end_line": 2},
        "distance": distance,
    }


@pytest.fixture
def stub(monkeypatch):
    """Swap all three agents. Each returns a canned answer and records its
    inputs, so a test can assert what an agent was actually shown."""
    seen = {"triage": [], "judge": [], "conventions": []}
    answers = {
        "triage": ("JUDGE", 0.30, [0], "candidate 0 looks equivalent"),
        "judge": ("DUPLICATE", "same arithmetic", "REPLACE", "call the existing one"),
        "conventions": ("FOLLOWS", "consistent with neighbours"),
    }

    def fake(kind):
        def inner(*args, **kwargs):
            seen[kind].append({"args": args, "kwargs": kwargs})
            return answers[kind]
        return inner

    monkeypatch.setattr("codebloatguard.triage.triage", fake("triage"))
    monkeypatch.setattr("codebloatguard.judge.judge", fake("judge"))
    monkeypatch.setattr("codebloatguard.conventions.check_conventions", fake("conventions"))

    return {"seen": seen, "answers": answers}


def run(store, code="def reduce_by_half(n):\n    return n / 2", **kwargs):
    return agent_module.review_chunk(
        store, code=code, label="newcode.py:1 reduce_by_half",
        vector=[0.1] * 8, **kwargs
    )


class TestHappyPath:
    def test_duplicate_survives_to_findings(self, stub):
        store = FakeStore([hit("one_half", "def one_half(x):\n    return x * 0.5", 0.1687)])
        state = run(store)

        assert len(state["findings"]) == 1
        assert state["findings"][0]["verdict"] == "DUPLICATE"

    def test_retrieval_triage_and_judge_all_run(self, stub):
        store = FakeStore([hit("one_half", "def one_half(x): return x * 0.5", 0.1687)])
        run(store)

        assert len(stub["seen"]["triage"]) == 1
        assert len(stub["seen"]["judge"]) == 1


class TestConventionsIsSkippedForDeletedCode:
    """Conventions advises on code someone keeps. A confirmed REPLACE says the
    function is going away, so the advice is noise and the call is wasted."""

    def test_replace_skips_conventions(self, stub):
        store = FakeStore([hit("one_half", "def one_half(x): return x * 0.5", 0.1687)])
        state = run(store)

        assert stub["seen"]["conventions"] == []
        assert "conventions" not in state

    @pytest.mark.parametrize("action", ["EXTRACT", "KEEP_BOTH"])
    def test_surviving_function_still_gets_conventions(self, stub, monkeypatch, action):
        monkeypatch.setattr(
            "codebloatguard.judge.judge",
            lambda *a, **k: ("DUPLICATE", "overlap", action, "pull out the shared part"),
        )
        store = FakeStore([hit("one_half", "def one_half(x): return x * 0.5", 0.1687)])
        run(store)

        assert len(stub["seen"]["conventions"]) == 1

    def test_judge_error_still_gets_conventions(self, stub, monkeypatch):
        """One dead call is not a reason to abandon a different question."""
        monkeypatch.setattr(
            "codebloatguard.judge.judge",
            lambda *a, **k: ("ERROR", "judge call failed: TimeoutError", "NONE", ""),
        )
        store = FakeStore([hit("one_half", "def one_half(x): return x * 0.5", 0.1687)])
        run(store)

        assert len(stub["seen"]["conventions"]) == 1


class TestAgentInputs:
    def test_judge_sees_the_pair_not_the_candidate_list(self, stub):
        """judge takes two code strings. Handing it the retrieval dict would
        embed metadata in the prompt and quietly change what is judged."""
        store = FakeStore([hit("one_half", "def one_half(x): return x * 0.5", 0.1687)])
        run(store)

        args = stub["seen"]["judge"][0]["args"]
        assert args[0].startswith("def reduce_by_half")
        assert args[1].startswith("def one_half")

    def test_conventions_sees_candidates_not_findings(self, stub, monkeypatch):
        """Precedent comes from the neighbours regardless of whether any of
        them turned out to be duplicates. Triage stops here, so there are no
        findings at all and conventions still has everything it needs."""
        monkeypatch.setattr(
            "codebloatguard.triage.triage",
            lambda *a, **k: ("STOP", 0.30, [], "nothing close enough"),
        )
        store = FakeStore([
            hit("one_half", "def one_half(x): return x * 0.5", 0.1687),
            hit("one_fifth", "def one_fifth(x): return x / 5", 0.34),
        ])
        run(store)

        neighbours = stub["seen"]["conventions"][0]["args"][1]
        assert {n["name"] for n in neighbours} == {"one_half", "one_fifth"}

    def test_conventions_is_capped_at_three_neighbours(self, stub, monkeypatch):
        monkeypatch.setattr(
            "codebloatguard.triage.triage",
            lambda *a, **k: ("STOP", 0.30, [], "nothing close enough"),
        )
        store = FakeStore([hit(f"f{i}", f"def f{i}(): pass", 0.2 + i / 100) for i in range(6)])
        run(store)

        assert len(stub["seen"]["conventions"][0]["args"][1]) == 3

    def test_exclude_paths_reaches_the_store(self, stub):
        store = FakeStore([hit("one_half", "def one_half(x): return x * 0.5", 0.1687)])
        run(store, exclude_paths={"newcode.py"})

        assert store.calls[0]["exclude_paths"] == {"newcode.py"}


class TestControlFlow:
    def test_stop_skips_the_judge_entirely(self, stub, monkeypatch):
        monkeypatch.setattr(
            "codebloatguard.triage.triage",
            lambda *a, **k: ("STOP", 0.30, [], "nearest candidate is unrelated"),
        )
        store = FakeStore([hit("unrelated", "def unrelated(): pass", 0.55)])
        state = run(store)

        assert state["findings"] == []
        assert stub["seen"]["judge"] == []
        assert len(stub["seen"]["conventions"]) == 1

    def test_widen_retrieves_again_with_a_larger_k(self, stub, monkeypatch):
        monkeypatch.setattr(
            "codebloatguard.triage.triage",
            lambda *a, **k: ("WIDEN", 0.45, [], "furthest candidate still plausible"),
        )
        store = FakeStore([hit("maybe", "def maybe(): pass", 0.42)])
        run(store)

        assert [c["k"] for c in store.calls] == [5, 10, 15]

    def test_widen_is_capped_at_max_attempts(self, stub, monkeypatch):
        monkeypatch.setattr(
            "codebloatguard.triage.triage",
            lambda *a, **k: ("WIDEN", 0.45, [], "keep going"),
        )
        store = FakeStore([hit("maybe", "def maybe(): pass", 0.42)])
        run(store)

        assert len(store.calls) == agent_module.MAX_ATTEMPTS

    def test_empty_retrieval_stops_without_calling_triage(self, stub):
        state = run(FakeStore([]))

        assert stub["seen"]["triage"] == []
        assert state["findings"] == []
        assert len(stub["seen"]["conventions"]) == 1


class TestTrace:
    def test_judging_path_is_recorded(self, stub):
        store = FakeStore([hit("one_half", "def one_half(x): return x * 0.5", 0.1687)])
        trace = " ".join(run(store)["trace"])

        for step in ("retrieve:", "triage:", "judge:"):
            assert step in trace

    def test_stopping_path_is_recorded(self, stub, monkeypatch):
        monkeypatch.setattr(
            "codebloatguard.triage.triage",
            lambda *a, **k: ("STOP", 0.30, [], "unrelated"),
        )
        store = FakeStore([hit("other", "def other(): pass", 0.55)])
        trace = " ".join(run(store)["trace"])

        for step in ("retrieve:", "triage:", "conventions:"):
            assert step in trace

    def test_trace_names_the_agent_that_ended_the_run(self, stub):
        """A REPLACE ends at judge. The trace should show that rather than
        looking like conventions was skipped by accident."""
        store = FakeStore([hit("one_half", "def one_half(x): return x * 0.5", 0.1687)])
        assert run(store)["trace"][-1].startswith("judge:")
