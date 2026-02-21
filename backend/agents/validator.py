"""Agent 6: Validator — deterministic pass/fail rubric.

The final gatekeeper. Runs all tests, captures after-metrics, and
uses the deterministic rubric from metrics_engine to produce a verdict.

NO LLM calls. Pure computation + test execution.

Owner: ___
Status: NOT STARTED
Depends on: core/metrics_engine.py, core/sandbox.py, models/types.py
"""

from __future__ import annotations

from backend.models.types import (
    BehavioralSnapshot,
    OptimizedCode,
    TestSuite,
    TestResults,
    ValidationResult,
)


def validator_agent(
    before_snapshot: BehavioralSnapshot,
    optimized: OptimizedCode,
    test_suite: TestSuite,
    existing_test_file: str | None = None,
) -> ValidationResult:
    """Validate an optimization with tests and deterministic rubric.

    Pipeline position: SIXTH (final stage)
    Input: before snapshot, optimized code, generated tests, existing tests
    Output: ValidationResult with APPROVED or REJECTED verdict

    Uses LLM: NO (pure deterministic)

    Owner: ___
    Status: NOT STARTED
    Depends on: core/metrics_engine.py, core/sandbox.py, models/types.py
    """
    # TODO: Step 1 — Run existing tests against optimized code
    #   - If existing_test_file exists, exec it with the optimized function
    #   - Record pass/fail
    # TODO: Step 2 — Run differential tests
    #   - Exec test_suite.differential_test_code
    #   - Record pass/fail and capture failure details
    # TODO: Step 3 — Run targeted tests
    #   - Exec test_suite.targeted_test_code
    #   - Record pass/fail and capture failure details
    # TODO: Step 4 — Capture after-snapshot
    #   - Run compute_static_metrics on optimized code
    #   - Benchmark the optimized function
    #   - Estimate Big O
    # TODO: Step 5 — Run deterministic rubric
    #   - Call compute_rubric(before_metrics, after_metrics, before_bench, after_bench, test_results)
    #   - This handles vetoes (test failures, regressions) and improvements
    # TODO: Step 6 — Return ValidationResult
    pass


def _run_tests(test_code: str, function_source: str) -> tuple[bool, list[str]]:
    """Execute test code against a function and return (passed, failures).

    Owner: ___
    Status: NOT STARTED
    Depends on: core/sandbox.py
    """
    # TODO: Set up a temp directory with the test file and function file
    # TODO: Run pytest programmatically or via subprocess
    # TODO: Parse results for pass/fail
    # TODO: Return (all_passed, list_of_failure_messages)
    pass
