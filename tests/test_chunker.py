"""
Chunker tests. No API key, no network, no store.

Everything downstream inherits whatever this produces: a function the chunker
misses is invisible to retrieval, and two chunks sharing an id mean one of
them is silently dropped before it is ever embedded. So the cases that matter
here are the ones indexing psf/requests turned up, all of which were quiet
wrong answers rather than crashes.
"""

from pathlib import Path

import pytest

from codebloatguard.indexing.chunker import chunk_file, chunk_repo


def names(chunks) -> set[str]:
    return {c.name for c in chunks}


def ids(chunks) -> list[str]:
    return [c.id for c in chunks]


class TestExtraction:
    def test_finds_module_level_functions(self, sample_repo):
        found = names(chunk_file(sample_repo / "geometry.py", sample_repo))
        assert {"triangle_area", "rectangle_area", "circle_area", "halve"} <= found

    def test_classes_are_scope_not_chunks(self, sample_repo):
        """A class is a container. Indexing one would embed every method
        concatenated, which retrieves on the union of things it does and
        matches nothing precisely."""
        chunks = chunk_file(sample_repo / "geometry.py", sample_repo)
        assert "Rectangle" not in names(chunks)
        assert ("area", "Rectangle") in {(c.name, c.parent) for c in chunks}

    def test_nested_functions_are_chunked(self, sample_repo):
        chunks = chunk_file(sample_repo / "geometry.py", sample_repo)
        label = next(c for c in chunks if c.name == "label")
        assert label.parent == "describe_shapes"

    def test_line_numbers_point_at_the_definition(self, sample_repo):
        chunks = chunk_file(sample_repo / "geometry.py", sample_repo)
        triangle = next(c for c in chunks if c.name == "triangle_area")
        assert (triangle.start_line, triangle.end_line) == (1, 2)
        assert triangle.code.startswith("def triangle_area")

    def test_paths_are_relative_to_repo_root(self, sample_repo):
        chunks = chunk_file(sample_repo / "geometry.py", sample_repo)
        assert all(c.path == "geometry.py" for c in chunks)
        assert not any(Path(c.path).is_absolute() for c in chunks)


class TestOverloadStubs:
    """@overload stubs have a body of `...` and no logic. Indexing them made
    stub-vs-stub the top scoring "duplicate" pairs in the requests benchmark,
    because three empty bodies are of course identical to each other."""

    def test_only_the_implementation_is_chunked(self, edge_cases):
        chunks = chunk_file(edge_cases / "overload_stubs.py", edge_cases)
        assert len(chunks) == 1
        assert "isinstance" in chunks[0].code

    def test_stub_bodies_never_reach_the_index(self, edge_cases):
        chunks = chunk_file(edge_cases / "overload_stubs.py", edge_cases)
        assert not any(c.code.rstrip().endswith("...") for c in chunks)


class TestIdCollisions:
    """Ids are path + scoped name + content hash. Two chunks sharing one id
    is not an error anywhere: the store dedupes per file and the loser is
    dropped before embedding, so the only symptom is a function that silently
    stops being searchable."""

    def test_identical_methods_in_different_classes_stay_distinct(self, edge_cases):
        """Cart.__eq__ and Queue.__eq__ are byte-identical. Without the class
        in the id both hash to the same value and Chroma rejects the batch."""
        chunks = chunk_file(edge_cases / "twin_classes.py", edge_cases)
        assert len(ids(chunks)) == len(set(ids(chunks)))
        assert {"twin_classes.py:Cart.__eq__", "twin_classes.py:Queue.__eq__"} <= {
            c.id.rsplit(":", 1)[0] for c in chunks
        }

    def test_class_redefined_in_one_file_collides_on_purpose(self, edge_cases):
        """Same path, same scope, same name, same body. There is nothing left
        to tell them apart, and Python itself keeps only the last one, so the
        store dedupes rather than the chunker inventing a difference."""
        chunks = chunk_file(edge_cases / "redefined_class.py", edge_cases)
        assert len(chunks) == 2
        assert len(set(ids(chunks))) == 1

    def test_nested_function_keeps_its_class_scope(self, tmp_path):
        """Two classes, same method name, byte-identical nested helper. The
        scope has to carry every level: with only the immediate parent both
        helpers are 'totals.helper', hash alike, and the store keeps one."""
        src = tmp_path / "nested.py"
        src.write_text(
            "class Cart:\n"
            "    def totals(self):\n"
            "        def helper():\n"
            "            return 1\n"
            "        return helper()\n"
            "\n"
            "class Queue:\n"
            "    def totals(self):\n"
            "        def helper():\n"
            "            return 1\n"
            "        return helper()\n"
        )
        chunks = chunk_file(src, tmp_path)

        assert len(ids(chunks)) == len(set(ids(chunks)))
        assert {c.id.rsplit(":", 1)[0] for c in chunks} == {
            "nested.py:Cart.totals",
            "nested.py:Cart.totals.helper",
            "nested.py:Queue.totals",
            "nested.py:Queue.totals.helper",
        }

    def test_scope_survives_arbitrary_nesting_depth(self, tmp_path):
        src = tmp_path / "deep.py"
        src.write_text(
            "class Outer:\n"
            "    class Inner:\n"
            "        def method(self):\n"
            "            def helper():\n"
            "                return 1\n"
            "            return helper()\n"
        )
        chunks = chunk_file(src, tmp_path)

        assert "deep.py:Outer.Inner.method.helper" in {
            c.id.rsplit(":", 1)[0] for c in chunks
        }


class TestIdStability:
    """The id is what makes re-indexing cheap: unchanged code keeps its id and
    is never re-embedded. Both halves of that matter."""

    def test_same_code_same_id(self, tmp_path):
        (tmp_path / "a.py").write_text("def f():\n    return 1\n")
        first = chunk_file(tmp_path / "a.py", tmp_path)
        second = chunk_file(tmp_path / "a.py", tmp_path)
        assert ids(first) == ids(second)

    def test_moving_a_function_within_a_file_does_not_change_its_id(self, tmp_path):
        src = tmp_path / "a.py"
        src.write_text("def f():\n    return 1\n\n\ndef g():\n    return 2\n")
        before = {c.name: c.id for c in chunk_file(src, tmp_path)}

        src.write_text("def g():\n    return 2\n\n\ndef f():\n    return 1\n")
        after = {c.name: c.id for c in chunk_file(src, tmp_path)}

        assert before == after

    def test_editing_a_body_changes_only_that_id(self, tmp_path):
        src = tmp_path / "a.py"
        src.write_text("def f():\n    return 1\n\n\ndef g():\n    return 2\n")
        before = {c.name: c.id for c in chunk_file(src, tmp_path)}

        src.write_text("def f():\n    return 99\n\n\ndef g():\n    return 2\n")
        after = {c.name: c.id for c in chunk_file(src, tmp_path)}

        assert after["f"] != before["f"]
        assert after["g"] == before["g"]


class TestRepoWalk:
    def test_skips_configured_directories(self, tmp_path):
        (tmp_path / "real.py").write_text("def kept():\n    return 1\n")
        vendored = tmp_path / ".venv" / "lib"
        vendored.mkdir(parents=True)
        (vendored / "dep.py").write_text("def skipped():\n    return 1\n")

        assert names(chunk_repo(tmp_path)) == {"kept"}

    def test_unparseable_file_does_not_abort_the_walk(self, tmp_path):
        (tmp_path / "broken.py").write_bytes(b"\xff\xfe def not_utf8(")
        (tmp_path / "fine.py").write_text("def survivor():\n    return 1\n")

        assert "survivor" in names(chunk_repo(tmp_path))


class TestEmbeddingText:
    def test_header_carries_path_and_scope(self, sample_repo):
        chunks = chunk_file(sample_repo / "geometry.py", sample_repo)
        area = next(c for c in chunks if c.name == "area" and c.parent == "Rectangle")
        text = area.for_embedding()
        assert "# file: geometry.py" in text
        assert "# symbol: Rectangle.area" in text
        assert area.code in text
