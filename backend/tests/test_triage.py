"""Tests for agents/triage.py

Owner: ___
Status: NOT STARTED
Depends on: agents/triage.py, demo-repos/data-toolkit/
"""

import pytest

# TODO: from backend.agents.triage import triage_agent
# TODO: import the demo repo path


class TestTriageAgent:
    # TODO: test scans demo repo and finds all functions
    # TODO: test classifies transforms.py functions as Category A
    # TODO: test classifies io_handlers.py functions as Category B
    # TODO: test classifies api_client.py functions as Category B
    # TODO: test skips config.py trivial functions
    # TODO: test detects pytest as test framework
    # TODO: test finds test_transforms.py for transforms.py functions
    # TODO: test sorts Category A by CC descending
    pass


class TestHelpers:
    # TODO: test _find_python_files skips __pycache__
    # TODO: test _extract_functions gets correct source ranges
    # TODO: test _classify_function detects open() as side effect
    # TODO: test _classify_function detects torch as GPU dependency
    pass
