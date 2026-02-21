"""Tests for core/metrics_engine.py

Owner: ___
Status: NOT STARTED
Depends on: core/metrics_engine.py
"""

import pytest

# TODO: from backend.core.metrics_engine import (
#     compute_static_metrics,
#     compute_cognitive_complexity,
#     benchmark_function,
#     estimate_big_o,
#     compute_rubric,
# )

BUBBLE_SORT = """
def custom_sort(items):
    result = list(items)
    n = len(result)
    for i in range(n):
        for j in range(0, n - i - 1):
            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]
    return result
"""

SORTED_BUILTIN = "def custom_sort(items):\\n    return sorted(items)\\n"


class TestStaticMetrics:
    # TODO: test CC is in expected range (4-6 for bubble sort)
    # TODO: test MI is in expected range (40-70)
    # TODO: test halstead_bugs > 0
    # TODO: test function not found returns defaults
    # TODO: test empty source code
    pass


class TestCognitiveComplexity:
    # TODO: test bubble sort cog complexity (~5-8)
    # TODO: test simple function returns 0
    # TODO: test nested if/elif/else chain
    # TODO: test boolean operators
    pass


class TestBenchmark:
    # TODO: test returns correct number of data points
    # TODO: test time increases with input size for O(n^2)
    # TODO: test timeout stops long-running benchmarks
    # TODO: test custom input generator works
    pass


class TestBigO:
    # TODO: test bubble sort estimates O(n^2) or O(n^3) slope
    # TODO: test sorted() estimates O(n) or O(n log n)
    # TODO: test insufficient data returns "O(?)"
    pass


class TestRubric:
    # TODO: test APPROVED when sorted() replaces bubble sort
    # TODO: test REJECTED when tests fail
    # TODO: test REJECTED on runtime regression
    # TODO: test REJECTED on memory regression
    # TODO: test REJECTED when no improvements detected
    pass
