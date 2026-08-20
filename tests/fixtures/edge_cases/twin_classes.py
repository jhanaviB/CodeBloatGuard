"""Pattern from psf/requests auth.py, shrunk.

Two classes whose __eq__ bodies are byte-identical. Before scope went into
the chunk id, both produced the id  path:__eq__:<same hash>  and Chroma
rejected the batch with DuplicateIDError.

What CodeBloatGuard must do: give them distinct ids via the class name,
Cart.__eq__ and Queue.__eq__.
"""


class Cart:
    def __init__(self, items):
        self.items = items

    def __eq__(self, other):
        return self.items == other.items


class Queue:
    def __init__(self, items):
        self.items = items

    def __eq__(self, other):
        return self.items == other.items
