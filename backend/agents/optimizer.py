"""Agent 4: Optimizer — code generation from hypothesis (LLM).

Takes the analyst's hypothesis and generates the actual optimized code.
This agent focuses purely on code generation — no metrics, no testing.

Owner: ___
Status: NOT STARTED
Depends on: integrations/llm_client.py, models/types.py
"""

from __future__ import annotations

from backend.models.types import BehavioralSnapshot, AnalysisHypothesis, OptimizedCode


def optimizer_agent(
    snapshot: BehavioralSnapshot,
    hypothesis: AnalysisHypothesis,
) -> OptimizedCode:
    """Generate optimized code based on the analyst's hypothesis.

    Pipeline position: FOURTH (runs after analyst, before test_designer)
    Input: BehavioralSnapshot + AnalysisHypothesis
    Output: OptimizedCode with the rewritten function

    Uses LLM: YES (call_claude)

    Owner: ___
    Status: NOT STARTED
    Depends on: integrations/llm_client.py, models/types.py
    """
    # TODO: Build prompt with:
    #   - Original source code
    #   - The hypothesis (what to optimize and how)
    #   - Constraints: must preserve function signature, must handle same edge cases
    #   - Style guide: match existing code style, keep it readable
    # TODO: Ask Claude to generate ONLY the new function body
    # TODO: Validate that the generated code:
    #   - Has the same function name and signature
    #   - Actually compiles (try compile())
    #   - Doesn't import banned modules (os.system, subprocess, etc.)
    # TODO: Extract just the function source from Claude's response
    # TODO: Return OptimizedCode with original and optimized source
    pass
