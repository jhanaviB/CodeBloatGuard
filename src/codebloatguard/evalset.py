"""
Labeled pairs for scoring the judge. Run with `cbg eval`.

Every pair is chosen to be hard in a specific way, named in its id. The
DUPLICATE pairs share behaviour but not surface form, so grep and even
embeddings alone would not pair them confidently. The DIFFERENT pairs share
surface form but not behaviour, which is where a judge that reads vocabulary
instead of logic goes wrong. History earned some of these: an earlier judge
configuration called triangle_area a duplicate of a halving function.
"""

EVAL_PAIRS = [
    # DUPLICATE: same behaviour, different syntax
    {
        "id": "loop-vs-recursion",
        "expected": "DUPLICATE",
        "new": "def factorial(n):\n    r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r",
        "old": "def fact(x):\n    return 1 if x <= 1 else x * fact(x - 1)",
    },
    {
        "id": "loop-vs-builtin-sum",
        "expected": "DUPLICATE",
        "new": "def total(xs):\n    t = 0\n    for x in xs:\n        t += x\n    return t",
        "old": "def add_all(values):\n    return sum(values)",
    },
    {
        "id": "div-vs-mult",
        "expected": "DUPLICATE",
        "new": "def halve(x):\n    return x / 2",
        "old": "def divide_by_two(a):\n    return a * 0.5",
    },
    {
        "id": "get-vs-tryexcept",
        "expected": "DUPLICATE",
        "new": "def lookup(d, k, default=None):\n    try:\n        return d[k]\n    except KeyError:\n        return default",
        "old": "def fetch(mapping, key, fallback=None):\n    return mapping.get(key, fallback)",
    },
    {
        "id": "manual-max-vs-builtin",
        "expected": "DUPLICATE",
        "new": "def biggest(nums):\n    best = nums[0]\n    for n in nums[1:]:\n        if n > best:\n            best = n\n    return best",
        "old": "def largest(items):\n    return max(items)",
    },
    {
        "id": "comprehension-vs-loop",
        "expected": "DUPLICATE",
        "new": "def evens(xs):\n    out = []\n    for x in xs:\n        if x % 2 == 0:\n            out.append(x)\n    return out",
        "old": "def even_numbers(values):\n    return [v for v in values if v % 2 == 0]",
    },
    # SIMILAR: real overlap, not interchangeable
    {
        "id": "logging-wrapper",
        "expected": "SIMILAR",
        "new": "def parse_and_log(raw):\n    print(f'parsing {raw!r}')\n    value = int(raw.strip())\n    print(f'parsed {value}')\n    return value",
        "old": "def parse(raw):\n    return int(raw.strip())",
    },
    {
        "id": "superset-normalise",
        "expected": "SIMILAR",
        "new": "def normalise(s):\n    return s.strip().lower()",
        "old": "def lower(s):\n    return s.lower()",
    },
    # DIFFERENT: same shape, different behaviour
    {
        "id": "inverse-predicates",
        "expected": "DIFFERENT",
        "new": "def is_even(n):\n    return n % 2 == 0",
        "old": "def is_odd(n):\n    return n % 2 == 1",
    },
    {
        "id": "different-divisor",
        "expected": "DIFFERENT",
        "new": "def third(x):\n    return x / 3",
        "old": "def halve(x):\n    return x / 2",
    },
    {
        "id": "mean-vs-median",
        "expected": "DIFFERENT",
        "new": "def mean(xs):\n    return sum(xs) / len(xs)",
        "old": "def median(xs):\n    s = sorted(xs)\n    return s[len(s) // 2]",
    },
    {
        "id": "swapped-params",
        "expected": "DIFFERENT",
        "new": "def clamp(v, lo, hi):\n    return max(lo, min(hi, v))",
        "old": "def clamp(v, hi, lo):\n    return max(lo, min(hi, v))",
    },
    {
        "id": "sort-direction",
        "expected": "DIFFERENT",
        "new": "def rank(xs):\n    return sorted(xs)",
        "old": "def rank_desc(xs):\n    return sorted(xs, reverse=True)",
    },
    {
        "id": "off-by-one-range",
        "expected": "DIFFERENT",
        "new": "def indices(n):\n    return list(range(n))",
        "old": "def one_through(n):\n    return list(range(1, n + 1))",
    },
]
