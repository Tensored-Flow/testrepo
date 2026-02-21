"""Agent 3: Analyst — hypothesis formation and failure diagnosis (LLM).

Examines the behavioral snapshot and source code, then generates a
hypothesis about what's wrong and how to fix it. This is where the
LLM's reasoning ability is most critical.

Owner: ___
Status: NOT STARTED
Depends on: integrations/llm_client.py, models/types.py
"""

from __future__ import annotations

from backend.models.types import BehavioralSnapshot, AnalysisHypothesis


def analyst_agent(snapshot: BehavioralSnapshot) -> AnalysisHypothesis:
    """Analyze a function's snapshot and form an optimization hypothesis.

    Pipeline position: THIRD (runs after snapshot, before optimizer)
    Input: BehavioralSnapshot from the snapshot agent
    Output: AnalysisHypothesis with optimization strategy

    Uses LLM: YES (call_claude_json)

    Owner: ___
    Status: NOT STARTED
    Depends on: integrations/llm_client.py, models/types.py
    """
    # TODO: Build a detailed prompt including:
    #   - The source code
    #   - Static metrics (CC, cognitive complexity, Halstead)
    #   - Benchmark results and Big O estimate
    #   - Nesting depth and parameter count
    # TODO: Ask Claude to identify:
    #   - The algorithmic complexity class (brute force, redundant, etc.)
    #   - The specific bottleneck (which loop/operation is O(n^2))
    #   - A concrete optimization strategy
    #   - Expected improvement (proposed Big O)
    #   - Risks and edge cases to watch for
    # TODO: Parse JSON response into AnalysisHypothesis
    # TODO: Validate that the hypothesis is actionable (has specific approach)
    pass


def diagnose_failure(
    snapshot: BehavioralSnapshot,
    failed_optimization: str,
    failure_reason: str,
) -> AnalysisHypothesis:
    """Re-analyze after a failed optimization attempt.

    Called when the validator rejects an optimization. The analyst
    examines what went wrong and proposes a different approach.

    Uses LLM: YES

    Owner: ___
    Status: NOT STARTED
    Depends on: integrations/llm_client.py
    """
    # TODO: Include the failed code and failure reason in the prompt
    # TODO: Ask Claude to diagnose why the optimization broke correctness
    # TODO: Generate a new, more conservative hypothesis
    # TODO: Return revised AnalysisHypothesis
    pass
