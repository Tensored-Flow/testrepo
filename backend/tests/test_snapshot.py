"""Tests for agents/snapshot.py and core/behavioral_snapshot.py

Owner: ___
Status: NOT STARTED
Depends on: agents/snapshot.py, core/behavioral_snapshot.py, core/sandbox.py
"""

import pytest

# TODO: from backend.agents.snapshot import snapshot_agent
# TODO: from backend.core.behavioral_snapshot import capture_snapshot, compare_outputs
# TODO: from backend.models.types import TargetFunction, Category


SORT_SOURCE = """
def custom_sort(items):
    result = list(items)
    n = len(result)
    for i in range(n):
        for j in range(0, n - i - 1):
            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]
    return result
"""


class TestSnapshotAgent:
    # TODO: test captures static metrics
    # TODO: test captures benchmark data with timing > 0
    # TODO: test captures input/output pairs
    # TODO: test Big O estimate is populated
    # TODO: test snapshot is reproducible (run twice, same result)
    pass


class TestCompareOutputs:
    # TODO: test identical outputs return (True, [])
    # TODO: test different outputs return (False, [description])
    # TODO: test float comparison uses tolerance
    # TODO: test list comparison is element-wise
    pass
