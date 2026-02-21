"""Agent 5: Test Designer — diff-aware test generation (LLM).

Generates two kinds of tests:
1. Differential tests: run same inputs through old and new code, compare outputs
2. Targeted tests: edge cases specific to the optimization (e.g., empty input,
   large input, boundary conditions the new algorithm might miss)

Owner: ___
Status: NOT STARTED
Depends on: integrations/llm_client.py, models/types.py
"""

from __future__ import annotations

from backend.models.types import BehavioralSnapshot, OptimizedCode, TestSuite


def test_designer_agent(
    snapshot: BehavioralSnapshot,
    optimized: OptimizedCode,
) -> TestSuite:
    """Generate differential and targeted tests for an optimization.

    Pipeline position: FIFTH (runs after optimizer, before validator)
    Input: BehavioralSnapshot (original) + OptimizedCode
    Output: TestSuite with generated test code

    Uses LLM: YES (call_claude)

    Owner: ___
    Status: NOT STARTED
    Depends on: integrations/llm_client.py, models/types.py
    """
    # TODO: Generate differential tests:
    #   - Import both old and new function
    #   - Generate diverse inputs (edge cases, random, adversarial)
    #   - Assert old_func(input) == new_func(input) for all inputs
    #   - Use pytest parametrize for clean test structure
    # TODO: Generate targeted tests:
    #   - Analyze the diff between original and optimized
    #   - Identify edge cases the new algorithm might handle differently
    #   - Test boundary conditions (empty, single, very large, negative, etc.)
    #   - Test type handling (int vs float, unicode strings, etc.)
    # TODO: Validate generated tests actually compile (try compile())
    # TODO: Return TestSuite with both test types
    pass
