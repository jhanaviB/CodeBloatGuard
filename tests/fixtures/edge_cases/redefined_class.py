"""Pattern from psf/requests tests/test_requests.py (class BadFileObj), shrunk.

The SAME class defined twice in one file, identically. Scoped ids collide
anyway (same path, same scope, same name, same hash), so the store dedupes
per file before writing. Last definition wins, which is what Python itself
does with a redefinition.
"""


class Helper:
    def run(self):
        return 1


class Helper:  # noqa: F811  deliberate redefinition
    def run(self):
        return 1
