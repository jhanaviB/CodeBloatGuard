"""
Benchmark tests. No API calls, which is also the property under test.
"""

from codebloatguard.benchmark import measure, report, run


def write_repo(root, files: dict[str, str]):
    for name, body in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    return root


class TestMeasure:
    def test_counts_files_and_functions(self, tmp_path):
        write_repo(tmp_path, {
            "a.py": "def one():\n    return 1\n\n\ndef two():\n    return 2\n",
            "b.py": "def three():\n    return 3\n",
        })
        result = measure(tmp_path)
        assert result.files == 2
        assert result.functions == 3

    def test_reports_skipped_files_separately(self, tmp_path):
        """Vendored code is skipped, not parsed. Folding it into the file
        count would understate throughput on exactly the repos where the
        skip list earns its keep."""
        write_repo(tmp_path, {
            "real.py": "def kept():\n    return 1\n",
            ".venv/lib/dep.py": "def vendored():\n    return 1\n",
            "node_modules/x/y.py": "def also_vendored():\n    return 1\n",
        })
        result = measure(tmp_path)
        assert (result.files, result.skipped, result.functions) == (1, 2, 1)

    def test_counts_lines(self, tmp_path):
        write_repo(tmp_path, {"a.py": "def f():\n    return 1\n"})
        assert measure(tmp_path).lines == 2

    def test_records_elapsed_time(self, tmp_path):
        write_repo(tmp_path, {"a.py": "def f():\n    return 1\n"})
        assert measure(tmp_path).seconds > 0

    def test_empty_repo_does_not_divide_by_zero(self, tmp_path):
        result = measure(tmp_path)
        assert (result.files, result.functions) == (0, 0)
        assert result.functions_per_file == 0.0

    def test_names_the_repo_by_directory(self, tmp_path):
        assert measure(tmp_path).name == tmp_path.name


class TestRun:
    def test_measures_each_repo(self, tmp_path):
        for name in ("one", "two"):
            write_repo(tmp_path / name, {"a.py": "def f():\n    return 1\n"})
        assert len(run([tmp_path / "one", tmp_path / "two"])) == 2

    def test_missing_path_is_skipped_not_fatal(self, tmp_path):
        write_repo(tmp_path / "real", {"a.py": "def f():\n    return 1\n"})
        results = run([tmp_path / "real", tmp_path / "does_not_exist"])
        assert [r.name for r in results] == ["real"]


class TestReport:
    def test_includes_every_repo_and_a_total(self, tmp_path):
        for name in ("alpha", "beta"):
            write_repo(tmp_path / name, {"a.py": "def f():\n    return 1\n"})
        text = report(run([tmp_path / "alpha", tmp_path / "beta"]))
        assert "alpha" in text and "beta" in text and "total" in text

    def test_orders_by_functions_descending(self, tmp_path):
        write_repo(tmp_path / "small", {"a.py": "def f():\n    return 1\n"})
        write_repo(tmp_path / "big", {
            "a.py": "".join(f"def f{i}():\n    return {i}\n\n\n" for i in range(10))
        })
        lines = report(run([tmp_path / "small", tmp_path / "big"])).splitlines()
        body = [ln for ln in lines if ln.startswith(("small", "big"))]
        assert body[0].startswith("big")

    def test_empty_input_says_so(self):
        assert report([]) == "nothing measured"
