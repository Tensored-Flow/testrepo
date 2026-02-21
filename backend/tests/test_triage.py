"""Tests for agents/triage.py

Owner: ___
Status: NOT STARTED
Depends on: agents/triage.py, demo-repos/flask-data-toolkit/
"""

import pytest

# TODO: from backend.agents.triage import triage_agent
# TODO: import the demo repo path


class TestTriageAgent:
    # TODO: test scans demo repo and finds all functions
    # TODO: test classifies sort.py functions as Category A
    # TODO: test classifies handlers.py functions as Category B
    # TODO: test classifies train.py functions as Category B
    # TODO: test skips __init__.py trivial functions
    # TODO: test detects pytest as test framework
    # TODO: test finds test_sort.py for sort.py functions
    # TODO: test sorts Category A by CC descending
    pass


class TestHelpers:
    # TODO: test _find_python_files skips __pycache__
    # TODO: test _extract_functions gets correct source ranges
    # TODO: test _classify_function detects open() as side effect
    # TODO: test _classify_function detects torch as GPU dependency
    pass
