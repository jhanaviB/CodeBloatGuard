"""
Store tests. Chroma runs for real against a temp directory; only the embedder
is faked.

Faking the embedder rather than the store is deliberate. The bugs worth
catching here are Chroma's rules, not ours: it rejects a batch containing
duplicate ids, and it keeps whatever was written last. A fake collection would
enforce neither and the tests would pass while the tool broke.

The fake is a token frequency vector rather than a hash. A hash gives identical
text an identical vector but makes near-identical text arbitrarily far apart,
which is the one property a real embedder has and the store depends on.
"""

import hashlib
import re

import pytest

from codebloatguard.indexing.chunker import Chunk

DIMS = 64


def fake_vector(text: str) -> list[float]:
    """Bag of tokens hashed into fixed buckets. Shared vocabulary pulls two
    texts together the way a real embedding model would."""
    vec = [0.0] * DIMS
    for token in re.findall(r"\w+", text.lower()):
        vec[int(hashlib.sha256(token.encode()).hexdigest(), 16) % DIMS] += 1.0
    return vec or [0.0] * DIMS


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A RepoStore writing to an isolated Chroma directory, with embeddings
    stubbed. CHROMA_PATH is read at call time inside RepoStore.__init__, so
    patching the module attribute is enough."""
    monkeypatch.setattr("codebloatguard.config.CHROMA_PATH", tmp_path / "chroma")
    monkeypatch.setattr("codebloatguard.indexing.store.CHROMA_PATH", tmp_path / "chroma")
    monkeypatch.setattr(
        "codebloatguard.indexing.store.embed",
        lambda texts: [fake_vector(t) for t in texts],
    )
    monkeypatch.setattr(
        "codebloatguard.indexing.store.embed_one",
        lambda text: fake_vector(text),
    )

    from codebloatguard.indexing.store import RepoStore

    return RepoStore(tmp_path / "repo")


def chunk(name: str, code: str, path: str = "a.py", parent: str | None = None) -> Chunk:
    scope = f"{parent}." if parent else ""
    return Chunk(
        id=f"{path}:{scope}{name}:{hashlib.sha256(code.encode()).hexdigest()[:12]}",
        code=code,
        path=path,
        name=name,
        parent=parent,
        start_line=1,
        end_line=2,
    )


class TestSync:
    def test_writes_chunks(self, store):
        store.sync([chunk("f", "def f(): return 1"), chunk("g", "def g(): return 2")])
        assert store.count() == 2

    def test_is_idempotent(self, store):
        chunks = [chunk("f", "def f(): return 1")]
        store.sync(chunks)
        store.sync(chunks)
        store.sync(chunks)
        assert store.count() == 1

    def test_edited_function_replaces_the_old_one(self, store):
        store.sync([chunk("f", "def f(): return 1")])
        store.sync([chunk("f", "def f(): return 99")])

        assert store.count() == 1
        hit = store.search("def f(): return 99", k=1)[0]
        assert hit["code"] == "def f(): return 99"

    def test_deleted_function_is_dropped(self, store):
        store.sync([chunk("f", "def f(): pass"), chunk("g", "def g(): pass")])
        store.sync([chunk("f", "def f(): pass")])

        assert store.count() == 1
        assert store.search("def f(): pass", k=5)[0]["meta"]["name"] == "f"

    def test_deleted_file_is_dropped(self, store):
        """_sync_file only visits paths present in this run, so a file that
        vanished is never revisited. Without the explicit sweep its chunks
        stay searchable forever and the tool reports duplicates against code
        that no longer exists."""
        store.sync([chunk("f", "def f(): pass", path="a.py"),
                    chunk("g", "def g(): pass", path="b.py")])
        store.sync([chunk("f", "def f(): pass", path="a.py")])

        assert store.count() == 1

    def test_moving_a_function_between_files_does_not_duplicate_it(self, store):
        store.sync([chunk("f", "def f(): pass", path="a.py")])
        store.sync([chunk("f", "def f(): pass", path="b.py")])

        assert store.count() == 1
        assert store.search("def f(): pass", k=1)[0]["meta"]["path"] == "b.py"


class TestDuplicateIds:
    def test_duplicate_ids_in_one_batch_do_not_raise(self, store):
        """Chroma rejects any batch containing duplicate ids, upsert included.
        A file defining the same class twice produces exactly that, so the
        store dedupes before writing."""
        same = chunk("run", "def run(self): return 1", parent="Helper")
        store.sync([same, same])
        assert store.count() == 1

    def test_last_definition_wins(self, store):
        """Matches Python: a redefinition shadows the first."""
        first = chunk("run", "def run(self): return 1", parent="Helper")
        second = Chunk(**{**first.__dict__, "start_line": 40, "end_line": 41})
        store.sync([first, second])

        assert store.search("def run(self): return 1", k=1)[0]["meta"]["start_line"] == 40


class TestSearch:
    def test_finds_the_nearest_chunk(self, store):
        store.sync([chunk("f", "def f(): return 1"), chunk("g", "def g(): return 2")])
        assert store.search("def f(): return 1", k=1)[0]["meta"]["name"] == "f"

    def test_k_caps_the_result_count(self, store):
        store.sync([chunk(n, f"def {n}(): pass") for n in "abcde"])
        assert len(store.search("def a(): pass", k=3)) == 3

    def test_query_matching_the_stored_text_scores_zero_distance(self, store):
        """Documents are stored as for_embedding(), header included, so an
        exact match means querying that same text."""
        c = chunk("f", "def f(): return 1")
        store.sync([c])
        hit = store.search_vec(fake_vector(c.for_embedding()), k=1)[0]
        assert hit["distance"] == pytest.approx(0, abs=1e-6)

    def test_bare_snippet_query_is_not_scored_the_same_way(self, store):
        """search() embeds the raw snippet while documents carry a
        '# file:'/'# symbol:' header, so the two sides are not embedded
        alike. cbg check builds its own header-carrying vectors and is
        consistent; cbg search goes through here and is not, which means the
        two commands report different distances for the same pair. Thresholds
        calibrated with one do not transfer to the other."""
        c = chunk("f", "def f(): return 1")
        store.sync([c])

        headered = store.search_vec(fake_vector(c.for_embedding()), k=1)[0]["distance"]
        bare = store.search(c.code, k=1)[0]["distance"]

        assert bare > headered

    def test_exclude_paths_hides_the_file_under_review(self, store):
        """The caller already holds newer in-memory chunks for files it is
        reviewing. Without this a function reliably matches its own indexed
        copy and every check reports itself as a duplicate."""
        store.sync([chunk("f", "def f(): pass", path="a.py"),
                    chunk("g", "def g(): pass", path="b.py")])

        hits = store.search("def f(): pass", k=5, exclude_paths={"a.py"})
        assert {h["meta"]["path"] for h in hits} == {"b.py"}

    def test_result_carries_location_metadata(self, store):
        store.sync([chunk("f", "def f(): pass", path="pkg/a.py")])
        meta = store.search("def f(): pass", k=1)[0]["meta"]
        assert meta["path"] == "pkg/a.py"
        assert meta["name"] == "f"
        assert meta["start_line"] == 1


class TestReset:
    def test_reset_empties_the_collection(self, store):
        store.sync([chunk("f", "def f(): pass")])
        store.reset()
        assert store.count() == 0
