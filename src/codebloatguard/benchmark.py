import time
from dataclasses import dataclass
from pathlib import Path
from codebloatguard.indexing.chunker import chunk_file
from codebloatguard.config import SKIP_DIRS

@dataclass
class Result:
    name: str
    files: int
    functions: int
    skipped: int
    lines: int
    seconds: float

    @property
    def functions_per_file(self) -> float:
        return self.functions / self.files if self.files > 0 else 0.0

def measure(root: Path) -> Result:
    start = time.time()
    files = 0
    functions = 0
    skipped = 0
    lines = 0
    
    for path in root.rglob("*.py"):
        if SKIP_DIRS & set(path.parts):
            skipped += 1
            continue
        
        files += 1
        try:
            lines += len(path.read_text().splitlines())
            functions += len(chunk_file(path, root))
        except Exception:
            pass
            
    return Result(
        name=root.name,
        files=files,
        functions=functions,
        skipped=skipped,
        lines=lines,
        seconds=time.time() - start
    )

def run(repos: list[Path]) -> list[Result]:
    results = []
    for repo in repos:
        if repo.exists():
            results.append(measure(repo))
    return results

def report(results: list[Result]) -> str:
    if not results:
        return "nothing measured"
    
    results.sort(key=lambda r: r.functions, reverse=True)
    
    lines = []
    for r in results:
        lines.append(f"{r.name}: {r.functions} functions in {r.files} files ({r.functions_per_file:.1f} f/f), {r.lines} lines in {r.seconds:.2f}s")
    
    total_files = sum(r.files for r in results)
    total_functions = sum(r.functions for r in results)
    total_lines = sum(r.lines for r in results)
    total_seconds = sum(r.seconds for r in results)
    
    lines.append(f"total: {total_functions} functions in {total_files} files, {total_lines} lines in {total_seconds:.2f}s")
    
    return "\n".join(lines)
