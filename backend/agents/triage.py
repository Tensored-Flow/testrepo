"""Agent 1: Triage — repo scanner and function classifier.

Scans a repository, finds all Python functions, and classifies each as:
  Category A: Pure, benchmarkable (no side effects)
  Category B: Has side effects (DB, file I/O, network, GPU) — skip benchmarking
  Category C: Too trivial or not a candidate — skip entirely

This agent does NOT use the LLM. It's pure AST analysis + heuristics.

Owner: ___
Status: NOT STARTED
Depends on: core/metrics_engine.py, models/types.py
"""

from __future__ import annotations

from backend.models.types import TargetFunction, TriageResult, Category


def triage_agent(repo_path: str) -> TriageResult:
    """Scan a repository and classify all Python functions.

    Pipeline position: FIRST (entry point)
    Input: path to a git repo
    Output: TriageResult with all classified functions

    Owner: ___
    Status: NOT STARTED
    Depends on: models/types.py, core/metrics_engine.py
    """
    # TODO: Walk repo_path recursively for .py files (skip __pycache__, venv, .git)
    # TODO: For each .py file, parse AST to extract function definitions
    # TODO: For each function, classify into Category A/B/C:
    #   - Detect side effects: open(), print(), requests., session., torch., .to(device)
    #   - Detect I/O: file read/write, DB queries, network calls
    #   - Detect triviality: functions < 3 lines, __init__, __repr__, etc.
    # TODO: For Category A functions, run quick lizard CC scan
    # TODO: Sort Category A by CC descending (optimize worst first)
    # TODO: Detect test framework (pytest vs unittest) from repo structure
    # TODO: Find matching test files for each function
    # TODO: Return populated TriageResult
    pass


def _find_python_files(repo_path: str) -> list[str]:
    """Recursively find all .py files, skipping common ignore dirs.

    Owner: ___
    Status: NOT STARTED
    """
    # TODO: os.walk, skip __pycache__, venv, .git, node_modules, .tox
    # TODO: Return sorted list of absolute paths
    pass


def _extract_functions(file_path: str) -> list[TargetFunction]:
    """Parse a .py file and extract all function/method definitions.

    Owner: ___
    Status: NOT STARTED
    """
    # TODO: ast.parse the file
    # TODO: Walk AST for FunctionDef and AsyncFunctionDef
    # TODO: Extract source code via ast.get_source_segment or lineno slicing
    # TODO: Return list of TargetFunction (category TBD)
    pass


def _classify_function(func: TargetFunction) -> Category:
    """Classify a function as Category A, B, or C.

    Owner: ___
    Status: NOT STARTED
    """
    # TODO: Parse function body AST
    # TODO: Check for side-effect indicators:
    #   Category B signals: open(), session.query, requests.get, torch., .cuda(),
    #     subprocess., os.system, print() in non-debug context, file write
    # TODO: Check for triviality:
    #   Category C signals: < 3 NLOC, only return/pass, dunder methods,
    #     @property, config/constants
    # TODO: Default to Category A
    pass
