"""Pattern from psf/requests models.py, shrunk.

Four defs of the same name: three @overload type stubs (body is just ...)
and one real implementation. Only the last one is logic.

What CodeBloatGuard must do: chunk ONLY the real one. Before the fix the
chunker indexed all four, and stub-vs-stub pairs were the top "duplicates"
in the benchmark. Try it: cbg chunk tests/fixtures/edge_cases
"""

from typing import overload


@overload
def encode(data: str) -> str: ...


@overload
def encode(data: bytes) -> bytes: ...


@overload
def encode(data: list) -> str: ...


def encode(data):
    if isinstance(data, bytes):
        return data
    return str(data)
