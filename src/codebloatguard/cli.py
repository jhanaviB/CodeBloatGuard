import argparse
from pathlib import Path

from codebloatguard.config import DUP_DISTANCE, ESCALATE_DISTANCE
from codebloatguard.indexing.chunker import chunk_repo
from codebloatguard.llm import DailyQuotaExhausted

def cmd_chunk(args):
    """
    Chunks the entire repo
    Fails if you pass just a filename
    """
    chunks = chunk_repo(Path(args.repo).resolve())
    for c in chunks:
        scope = f"{c.parent}." if c.parent else ""
        print(f"{c.path}:{c.start_line}-{c.end_line}  {scope}{c.name}")
    print(f"\n{len(chunks)} chunks")
    if args.show and chunks:
        print("\n--- embedding text for first chunk ---")
        print(chunks[0].for_embedding())

def cmd_index(args):
    """
    Syncs the chunks in the store to what we have on the repo
    """
    from codebloatguard.indexing.store import RepoStore

    repo = Path(args.repo).resolve()
    say = (lambda *a: None) if args.quiet else print

    store = RepoStore(repo)
    if args.reset:
        store.reset()

    chunks = chunk_repo(repo)
    say(f"chunked {len(chunks)} functions")
    store.sync(chunks)
    say(f"indexed. collection holds {store.count()} chunks")


def _escalate(store, chunk, vector, pool_paths, max_distance) -> int:
    """
    Run when no nearest neighbours with DEDUP distance were found. Escalates up to ESCALATE_DISTANCE
    """
    from codebloatguard.agent import review_chunk

    state = review_chunk(
        store, chunk.code, chunk.name, vector,
        exclude_paths=pool_paths, max_distance=max_distance,
    )
    findings = state.get("findings", [])
    if not findings:
        return 0

    duplicates = 0
    print(f"\n{chunk.path}:{chunk.start_line}  {chunk.name}  (agent widened the search)")
    for f in findings:
        duplicates += f["verdict"] == "DUPLICATE"
        print(f"    {f['verdict']:<9} [{f['distance']:.4f}] {f['where']}")
        print(f"        {f['reason']}")
        if f.get("advice"):
            print(f"        {f['action']}: {f['advice']}")
    return duplicates


def check_chunks(chunks, vectors, store, k, max_distance, no_judge,
                 no_advice=True, escalate=True) -> tuple[int, int]:
    """
    Compare a pool of functions against the store and against each other.
    Pass 1 Search for similar vectors in the store, excluding the files passed to be checked.
    Pass 2 Searches for similarity with the rest of the pool.
    """
    from codebloatguard.indexing.embedder import cosine_distance
    from codebloatguard.judge import judge

    pool_paths = {c.path for c in chunks}
    duplicates = errors = 0

    for i, c in enumerate(chunks):
        # Search for similar vectors in the store, excluding the files passed to be checked
        candidates = [
            {
                "code": h["code"],
                "distance": h["distance"],
                "name": h["meta"]["name"],
                "where": f"{h['meta']['path']}:{h['meta']['start_line']}  {h['meta']['name']}",
            }
            for h in store.search_vec(vectors[i], k=k, exclude_paths=pool_paths)
        ]

        # Searches for similarity with the rest of the pool.
        # We skip this in phase 1 to get current chunks of files and not compare with redundant old chunks
        candidates += [
            {
                "code": o.code,
                "distance": cosine_distance(vectors[i], vectors[j]),
                "name": o.name,
                "where": f"{o.path}:{o.start_line}  {o.name}  "
                         + ("(same file)" if o.path == c.path else "(also changed)"),
            }
            for j, o in enumerate(chunks[i + 1:], start=i + 1)
        ]

        hits = sorted(
            (h for h in candidates if h["distance"] <= max_distance),
            key=lambda h: h["distance"],
        )
        if not hits:
            # If no hits are found escalating this decision to the agent. ESCALATE_DISTANCE set at 0.45
            nearest = min((h["distance"] for h in candidates), default=1.0)
            if escalate and not no_judge and max_distance < nearest <= ESCALATE_DISTANCE:
                duplicates += _escalate(store, c, vectors[i], pool_paths, max_distance)
            continue

        print(f"\n{c.path}:{c.start_line}  {c.name}")
        for h in hits:
            if no_judge:
                print(f"    [{h['distance']:.4f}] {h['where']}")
                continue

            verdict, reason, action, advice = judge(
                c.code, h["code"], new_label=c.name, old_label=h["name"]
            )
            if verdict == "DIFFERENT":  # the only verdict not worth showing
                continue
            duplicates += verdict == "DUPLICATE"
            errors += verdict == "ERROR"  # shown, but must not fail the build
            print(f"    {verdict:<9} [{h['distance']:.4f}] {h['where']}")
            print(f"        {reason}")
            if advice and not no_advice:
                print(f"        {action}: {advice}")

    return duplicates, errors


def _load_pool(paths: list[Path], repo: Path):
    """
    Chunk every file under review, then embed the lot in one batched call.
    """
    from codebloatguard.indexing.chunker import chunk_file
    from codebloatguard.indexing.embedder import embed

    chunks = []
    for p in paths:
        try:
            chunks += chunk_file(p, repo)
        except ValueError:
            raise SystemExit(f"{p} is not inside {repo} (pass --repo)")
    if not chunks:
        return [], []
    return chunks, embed([c.for_embedding() for c in chunks])


def _report(duplicates: int, errors: int, no_judge: bool):
    if no_judge:
        return
    print(f"\n{duplicates} duplicate(s)" + (f", {errors} judge error(s)" if errors else ""))
    if duplicates:
        raise SystemExit(1)


def _open_store(repo: Path):
    from codebloatguard.indexing.store import RepoStore

    store = RepoStore(repo)
    if store.count() == 0:
        raise SystemExit(f"nothing indexed yet. run: cbg index {repo}")
    return store

def cmd_check(args):
    """
    Flag code in one file that already exists elsewhere in the repo.
    Parameters to k, update max_distance, judge, and give advice
    """
    repo = Path(args.repo).resolve()
    store = _open_store(repo)

    chunks, vectors = _load_pool([Path(args.file).resolve()], repo)
    duplicates, errors = check_chunks(
        chunks, vectors, store, args.k, args.max_distance, args.no_judge, args.no_advice
    )
    _report(duplicates, errors, args.no_judge)

def cmd_review(args):
    """Agentic review of one file.

    Same inputs as `check`. The difference is that triage decides per function
    whether to judge the candidates, search wider, or stop. Stopping early
    costs less than `check`, widening costs more.
    """
    from codebloatguard.agent import review_chunk

    repo = Path(args.repo).resolve()
    store = _open_store(repo)
    chunks, vectors = _load_pool([Path(args.file).resolve()], repo)
    if not chunks:
        print("no functions found")
        return

    pool_paths = {c.path for c in chunks}
    duplicates = 0

    for i, c in enumerate(chunks):
        state = review_chunk(
            store, c.code, c.name, vectors[i],
            exclude_paths=pool_paths, max_distance=args.max_distance,
        )
        print(f"\n{c.path}:{c.start_line}  {c.name}")
        for line in state.get("trace", []):
            # Failed calls are always shown. A silent run and a rate limited
            # one must never look the same.
            if args.trace or "ERROR" in line:
                print(f"      {line}")

        for f in state.get("findings", []):
            duplicates += f["verdict"] == "DUPLICATE"
            print(f"    {f['verdict']:<9} [{f['distance']:.4f}] {f['where']}")
            print(f"        {f['reason']}")
            if f.get("advice"):
                print(f"        {f['action']}: {f['advice']}")

        verdict, notes = state.get("conventions", ("", ""))
        if verdict == "DEVIATES":
            print(f"    CONVENTION  {notes}")

    print(f"\n{duplicates} duplicate(s)")
    if duplicates:
        raise SystemExit(1)


def _changed_py_files(repo: Path, base: str) -> list[str]:
    """
    base...HEAD Compares the current branch with base.
    Doesn't compare the files that may have been added to base at the time
    """
    import subprocess

    def git(*args):
        r = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True
        )
        return r.stdout if r.returncode == 0 else None

    out = git("diff", "--name-only", "--diff-filter=d", f"origin/{base}...HEAD")
    if out is None:
        out = git("diff", "--name-only", "--diff-filter=d", f"{base}...HEAD")
    if out is None:
        raise SystemExit(
            f"could not diff against '{base}'. In CI, checkout with "
            f"fetch-depth: 0 so the base branch is available."
        )

    return [
        p for p in out.splitlines()
        if p.endswith(".py") and (repo / p).is_file()
    ]


def cmd_check_pr(args):
    """
    Check only the files a PR changed, against the indexed baseline (main) in the store.
    """
    import os

    repo = Path(args.repo).resolve()
    base = args.base or os.environ.get("GITHUB_BASE_REF") or "main"

    files = _changed_py_files(repo, base)
    if not files:
        print(f"no changed .py files vs {base}")
        return

    print(f"checking {len(files)} changed file(s) vs {base}")
    store = _open_store(repo)

    chunks, vectors = _load_pool([(repo / rel).resolve() for rel in files], repo)
    duplicates, errors = check_chunks(
        chunks, vectors, store, args.k, args.max_distance, args.no_judge,
        args.no_advice, escalate=not args.no_escalate,
    )
    _report(duplicates, errors, args.no_judge)

def cmd_search(args):
    """
    Embeds code or file passed in args and then compares it with the repo
    """
    from codebloatguard.indexing.store import RepoStore

    code = Path(args.file).read_text() if args.file else args.code
    print(f"QUERY:\n{code}\n")
    for r in RepoStore(Path(args.repo).resolve()).search(code, k=args.k):
        m = r["meta"]
        print(f"[{r['distance']:.4f}] {m['path']}:{m['start_line']}  {m['name']}")
        if args.show:
            print("    " + "\n    ".join(r["code"].splitlines()[:6]) + "\n")


def cmd_stats(args):
    """
    Count of chunks indexed
    """
    from codebloatguard.indexing.store import RepoStore

    print(f"{RepoStore(Path(args.repo).resolve()).count()} chunks indexed")


def cmd_eval(args):
    """
    Score the judge against the labeled pairs in evalset.py.

    Prints a per-pair result and a confusion matrix. With tracing on, every
    call lands in LangSmith, so a failing pair can be inspected there.
    --upload pushes the dataset itself to LangSmith for the evals UI.
    """
    from codebloatguard.evalset import EVAL_PAIRS
    from codebloatguard.judge import VERDICTS, judge

    results = []
    for p in EVAL_PAIRS:
        verdict, reason, _, _ = judge(p["new"], p["old"])
        ok = verdict == p["expected"]
        results.append((p, verdict, ok))
        mark = "PASS" if ok else "FAIL"
        print(f"{mark}  {p['id']:<24} expected {p['expected']:<9} got {verdict}")
        if not ok:
            print(f"      {reason}")

    labels = list(VERDICTS) + ["ERROR"]
    matrix = {e: {g: 0 for g in labels} for e in VERDICTS}
    for p, got, _ in results:
        matrix[p["expected"]][got if got in labels else "ERROR"] += 1

    print("\nexpected \\ got  " + "".join(f"{l:<11}" for l in labels))
    for e in VERDICTS:
        print(f"{e:<15} " + "".join(f"{matrix[e][g]:<11}" for g in labels))

    passed = sum(ok for _, _, ok in results)
    print(f"\n{passed}/{len(results)} correct")

    if args.upload:
        _upload_dataset(EVAL_PAIRS)


def _upload_dataset(pairs):
    """Push the labeled pairs to LangSmith as a dataset."""
    import os

    if not os.environ.get("LANGSMITH_API_KEY"):
        raise SystemExit("set LANGSMITH_API_KEY to upload")
    from langsmith import Client

    client = Client()
    name = "codebloatguard-judge"
    try:
        ds = client.read_dataset(dataset_name=name)
    except Exception:
        ds = client.create_dataset(dataset_name=name)
    client.create_examples(
        dataset_id=ds.id,
        inputs=[{"new_code": p["new"], "old_code": p["old"]} for p in pairs],
        outputs=[{"verdict": p["expected"]} for p in pairs],
    )
    print(f"uploaded {len(pairs)} examples to dataset '{name}'")


def cmd_judge(args):
    """
    Judge two snippets directly. Debug tool for prompt and effort tuning.
    """
    from codebloatguard.judge import judge

    verdict, reason, action, advice = judge(args.code_a, args.code_b)
    print(f"{verdict}: {reason}")
    if advice:
        print(f"{action}: {advice}")


def cmd_benchmark(args):
    """Parse throughput across repositories. Costs nothing and needs no key,
    so the numbers in the README are reproducible by anyone who clones it."""
    from codebloatguard.benchmark import report, run

    print(report(run([Path(r).resolve() for r in args.repos])))


def main():
    p = argparse.ArgumentParser(prog="cbg")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("chunk", help="show chunks without embedding")
    c.add_argument("repo")
    c.add_argument("--show", action="store_true", help="print first chunk's embedding text")
    c.set_defaults(func=cmd_chunk)

    i = sub.add_parser("index", help="converge the store to the disk")
    i.add_argument("repo")
    i.add_argument("--reset", action="store_true", help="drop the collection first")
    i.add_argument("--quiet", action="store_true", help="silence output")
    i.set_defaults(func=cmd_index)

    j = sub.add_parser("judge", help="verdict on similarity of two pieces of code")
    j.add_argument("code_a", help="1st piece of code")
    j.add_argument("code_b", help="2nd piece of code")
    j.set_defaults(func=cmd_judge)
    
    ch = sub.add_parser("check", help="flag code in a file that already exists elsewhere")
    ch.add_argument("file")
    ch.add_argument("--repo", default=".", help="repo root the index was built from")
    ch.add_argument("-k", type=int, default=3, help="candidates retrieved per function")
    ch.add_argument("--max-distance", type=float, default=0.1,
                    help="skip candidates further than this")
    ch.add_argument("--no-judge", action="store_true",
                    help="print candidates and distances without calling the judge")
    ch.add_argument("--no-advice", action="store_true",
                    help="stop after the verdict, skip the refactor suggestion")
    ch.set_defaults(func=cmd_check)

    rv = sub.add_parser("review", help="agentic review: the model picks the search radius")
    rv.add_argument("file")
    rv.add_argument("--repo", default=".", help="repo root the index was built from")
    rv.add_argument("--max-distance", type=float, default=DUP_DISTANCE,
                    help="starting radius, which the agent may widen")
    rv.add_argument("--trace", action="store_true", help="print each step the agent took")
    rv.set_defaults(func=cmd_review)

    cp = sub.add_parser("check-pr", help="check only the files a PR changed")
    cp.add_argument("--repo", default=".", help="repo root the index was built from")
    cp.add_argument("--base", default=None, help="base branch (default: $GITHUB_BASE_REF or main)")
    cp.add_argument("-k", type=int, default=3, help="candidates retrieved per function")
    cp.add_argument("--max-distance", type=float, default=DUP_DISTANCE,
                    help="skip candidates further than this")
    cp.add_argument("--no-judge", action="store_true",
                    help="print candidates and distances without calling the judge")
    cp.add_argument("--no-advice", action="store_true",
                    help="stop after the verdict, skip the refactor suggestion")
    cp.add_argument("--no-escalate", action="store_true",
                    help=f"never hand near misses (under {ESCALATE_DISTANCE}) to the agent")
    cp.set_defaults(func=cmd_check_pr)

    s = sub.add_parser("search", help="retrieve nearest chunks")
    s.add_argument("code", nargs="?", help="code snippet")
    s.add_argument("--file", help="read snippet from file instead")
    s.add_argument("--repo", default=".", help="repo whose index to search")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--show", action="store_true", help="print matched code")
    s.set_defaults(func=cmd_search)

    st = sub.add_parser("stats", help="collection size")
    st.add_argument("--repo", default=".", help="repo whose index to inspect")
    st.set_defaults(func=cmd_stats)

    ev = sub.add_parser("eval", help="score the judge against labeled pairs")
    ev.add_argument("--upload", action="store_true",
                    help="also push the dataset to LangSmith")
    ev.set_defaults(func=cmd_eval)

    bm = sub.add_parser("benchmark", help="chunker throughput on real repos, no API calls")
    bm.add_argument("repos", nargs="+", help="paths to checked-out repositories")
    bm.set_defaults(func=cmd_benchmark)

    args = p.parse_args()
    try:
        args.func(args)
    except DailyQuotaExhausted as e:
        # Exit 2, not 1. Exit 1 means duplicates were found, and a run that
        # never finished checking must not be reported as one that finished
        # and found nothing.
        raise SystemExit(f"\n{e}") from None


if __name__ == "__main__":
    main()
