import hashlib
from dataclasses import dataclass
from pathlib import Path

import tree_sitter_python
from tree_sitter import Language, Parser

from codebloatguard.config import SKIP_DIRS

PY = Language(tree_sitter_python.language())
TARGET_NODES = {"function_definition", "class_definition"}


@dataclass
class Chunk:
    id: str
    code: str
    path: str
    name: str
    parent: str | None
    start_line: int
    end_line: int

    def for_embedding(self) -> str:
        """Context header + code. The header is what makes retrieval work."""
        scope = f"{self.parent}." if self.parent else ""
        return f"# file: {self.path}\n# symbol: {scope}{self.name}\n{self.code}"


def _walk(node, source: bytes, path: str, parent: str | None, out: list[Chunk]):
    """
    Does a recursion through the program to find function_definition and class_definition.
    function_definition becomes a chunk
    class_definition becomes a parent and never a chunk on it's own.
    """
    for child in node.children:
        if child.type in TARGET_NODES:
            name_node = child.child_by_field_name("name")
            name = source[name_node.start_byte:name_node.end_byte].decode() if name_node else "<anon>"
            code = source[child.start_byte:child.end_byte].decode()
            scope = f"{parent}." if parent else ""

            if child.type == "function_definition" and not code.rstrip().endswith("..."):
                # endswith("...") is for ignoring overload for type declarations.
                out.append(Chunk(
                    id=f"{path}:{scope}{name}:{hashlib.sha256(code.encode()).hexdigest()[:12]}",
                    code=code,
                    path=path,
                    name=name,
                    parent=parent,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                ))

            _walk(child, source, path, f"{scope}{name}", out)
        else:
            _walk(child, source, path, parent, out)

def chunk_file(path: Path, repo_root: Path) -> list[Chunk]:
    source = path.read_bytes()
    tree = Parser(PY).parse(source)
    out: list[Chunk] = []
    _walk(tree.root_node, source, str(path.relative_to(repo_root)), None, out)
    return out


def chunk_repo(repo_root: Path) -> list[Chunk]:
    """
    Searched for all paths in repo_root. 
    Skips the path if any part of it is to be skipped
    Chunks file-by-file and adds it to the chunks list
    """
    chunks = []
    for path in repo_root.rglob("*.py"):
        if SKIP_DIRS & set(path.parts):
            continue
        try:
            chunks.extend(chunk_file(path, repo_root))
        except Exception as e:
            print(f"skip {path}: {e}")
    return chunks
