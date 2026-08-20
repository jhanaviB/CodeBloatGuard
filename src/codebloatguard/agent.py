from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from codebloatguard.config import WIDEN_STEP

MAX_ATTEMPTS = 3
DISTANCE_CEILING = 0.60
K_BASE = 5

MAX_PARALLEL = 4

class ReviewState(TypedDict, total=False):
    code: str
    label: str
    vector: list[float]
    exclude_paths: set[str]

    max_distance: float
    k: int
    attempts: int

    candidates: list[dict]   
    selected: list[dict]    
    decision: str
    trace: list[str]         

    findings: list[dict]
    conventions: tuple[str, str]

def _summarise(candidates: list[dict], limit: int = 8) -> str:
    lines = []
    for i, c in enumerate(candidates):
        head = "\n".join(c["code"].splitlines()[:limit])
        lines.append(f"[{i}] {c['name']}  distance={c['distance']:.4f}\n{head}")
    return "\n\n".join(lines)

def build_reviewer(store):
    """
    Builds the iterative worklfow and the state graph
    """
    from codebloatguard.conventions import check_conventions
    from codebloatguard.judge import judge
    from codebloatguard.triage import triage

    def retrieve(state: ReviewState) -> dict:
        attempts = state.get("attempts", 0) + 1
        k = state.get("k", K_BASE)
        hits = store.search_vec(
            state["vector"], k=k, exclude_paths=state.get("exclude_paths")
        )
        candidates = [
            {
                "code": h["code"],
                "distance": h["distance"],
                "name": h["meta"]["name"],
                "where": f"{h['meta']['path']}:{h['meta']['start_line']}  {h['meta']['name']}",
            }
            for h in hits
        ]

        #Counts how many candidates are within the distance threshold
        near = sum(c["distance"] <= state["max_distance"] for c in candidates)

        return {
            "candidates": candidates,
            "attempts": attempts,
            "trace": state.get("trace", [])
            + [f"retrieve: k={k} radius={state['max_distance']:.2f} "
               f"got {len(candidates)}, {near} inside radius"],
        }

    def triage_node(state: ReviewState) -> dict:
        if not state["candidates"]:
            return {
                "decision": "STOP",
                "selected": [],
                "trace": state.get("trace", []) + ["triage: nothing retrieved, stop"],
            }

        action, new_distance, keep, reason = triage(
            state["code"],
            _summarise(state["candidates"]),
            len(state["candidates"]),
            state["max_distance"],
            state.get("attempts", 1),
            MAX_ATTEMPTS,
        )

        if action == "WIDEN":
            if state.get("attempts", 1) >= MAX_ATTEMPTS:
                action, reason = "STOP", f"attempt cap reached ({reason})"
            else:
                new_distance = min(
                    max(new_distance, state["max_distance"] + WIDEN_STEP), DISTANCE_CEILING
                )

        selected = [state["candidates"][i] for i in keep if 0 <= i < len(state["candidates"])]
        if action == "JUDGE" and not selected:
            action, reason = "STOP", "triage selected nothing"

        return {
            "decision": action,
            "selected": selected,
            "max_distance": new_distance if action == "WIDEN" else state["max_distance"],
            "k": state.get("k", K_BASE) + K_BASE if action == "WIDEN" else state.get("k", K_BASE),
            "trace": state.get("trace", [])
            + [f"triage: {action} ({reason})"],
        }

    def judge_node(state: ReviewState) -> dict:
        """
        Concurrent check for code pairs
        Returns verdict and action. 
        Pairs that are "different" are dropped here
        """
        selected = state["selected"]
        if not selected:
            return {
                "findings": [],
                "trace": state.get("trace", []) + ["judge: nothing selected"],
            }

        def judge_one(c: dict) -> dict:
            verdict, reason, action, advice = judge(
                state["code"], c["code"], new_label=state["label"], old_label=c["name"]
            )
            return {**c, "verdict": verdict, "reason": reason,
                    "action": action, "advice": advice}

        with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL, len(selected))) as pool:
            judged = list(pool.map(judge_one, selected))

        findings = [f for f in judged if f["verdict"] != "DIFFERENT"]
        return {
            "findings": findings,
            "trace": state.get("trace", [])
            + [f"judge: {len(selected)} pairs in parallel, {len(findings)} flagged"],
        }

    def conventions_node(state: ReviewState) -> dict:
        """
        Checks new function with pre-existing code candidates to ascertain if conventions are followed.
        """
        verdict, notes = check_conventions(
            state["code"], state.get("candidates", [])[:3], new_label=state["label"]
        )
        return {
            "conventions": (verdict, notes),
            "trace": state.get("trace", []) + [f"conventions: {verdict}"],
        }

    def after_triage(state: ReviewState) -> str:
        return {"WIDEN": "retrieve", "JUDGE": "judge"}.get(state["decision"], "conventions")

    def after_judge(state: ReviewState) -> str:
        """Skip conventions when the function is not going to exist.

        Conventions reports on naming, argument order and return shape. 
        We still want to check for EXTRACT and DUPLICATE
        """
        replaced = any(
            f["verdict"] == "DUPLICATE" and f["action"] == "REPLACE"
            for f in state.get("findings", [])
        )
        return "end" if replaced else "conventions"

    g = StateGraph(ReviewState)
    g.add_node("retrieve", retrieve)
    g.add_node("triage", triage_node)
    g.add_node("judge", judge_node)
    g.add_node("conventions", conventions_node)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "triage")
    g.add_conditional_edges("triage", after_triage, ["retrieve", "judge", "conventions"])
    g.add_conditional_edges("judge", after_judge, {"conventions": "conventions", "end": END})
    g.add_edge("conventions", END)

    return g.compile()


def review_chunk(store, code: str, label: str, vector: list[float],
                 exclude_paths=None, max_distance: float = 0.30) -> dict:
    """Runs the graph over one function. Returns the final state."""
    return build_reviewer(store).invoke(
        {
            "code": code,
            "label": label,
            "vector": vector,
            "exclude_paths": exclude_paths,
            "max_distance": max_distance,
            "k": K_BASE,
            "attempts": 0,
            "trace": [],
            "findings": [],
        }
    )
