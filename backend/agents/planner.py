"""
Agent 0: Planner — Strategic Optimization Planning.

Uses Claude Haiku for FAST strategic decisions:
  - Which functions to optimize first (not just CC order)
  - Which functions might benefit from similar strategies
  - How to allocate the optimization budget

This adds a layer of strategic reasoning before the pipeline starts.
Demonstrates multi-model usage (Haiku for planning, Sonnet for execution).

Now writes plan_directives to ConversationContext for downstream agents.
"""

import json
from typing import Optional

from backend.models.types import TargetFunction
from backend.models.events import EventEmitter, EventType
from backend.integrations.llm_client import call_haiku_json
from backend.core.conversation import ConversationContext


PLANNER_SYSTEM = """\
You are a strategic code optimization planner. You receive a list of Python functions
that have been identified as optimization targets.

Your job: analyze them and create an execution plan that maximizes impact.

Consider:
1. PRIORITY: Which functions benefit most from optimization? (high CC + frequently called > low CC)
2. GROUPING: Which functions might use similar optimization strategies? (e.g., all sorting functions)
3. DEPENDENCIES: If function A calls function B, optimize B first
4. QUICK WINS: Low-hanging fruit that can be optimized fast vs complex rewrites
5. BUDGET: If we can only optimize N functions, which N give the most value?

Respond with JSON:
{
  "execution_order": [
    {
      "function_name": "...",
      "file_path": "...",
      "priority": "high/medium/low",
      "reason": "why this order",
      "suggested_strategy_hint": "brief hint for the Analyst",
      "group": "optional group name for similar functions",
      "estimated_difficulty": "easy/medium/hard"
    }
  ],
  "groups": [
    {
      "name": "sorting_functions",
      "strategy_hint": "All may benefit from divide-and-conquer approaches",
      "members": ["bubble_sort", "selection_sort"]
    }
  ],
  "estimated_quick_wins": 2,
  "planning_notes": "brief strategic notes"
}
"""


def planner_agent(
    targets: list[TargetFunction],
    max_functions: int = 10,
    emitter: Optional[EventEmitter] = None,
    ctx: Optional[ConversationContext] = None,
) -> dict:
    """
    Strategic planning agent. Uses Haiku for fast decisions.

    Returns an execution plan with ordered targets and groupings.
    Optionally writes plan_directives to ConversationContext for each target.

    Args:
        targets: List of target functions from triage
        max_functions: Maximum number of functions to optimize
        emitter: Event emitter for SSE streaming
        ctx: Optional ConversationContext (not per-function; used for global plan)
    """
    _log = lambda msg: emitter.log("planner", msg) if emitter else None

    _log(f"Planning optimization strategy for {len(targets)} functions...")

    # Build the function summary for the planner
    function_summaries = []
    for t in targets:
        lines = t.source_code.strip().split("\n")
        preview = "\n".join(lines[:3])
        if len(lines) > 4:
            preview += f"\n    ... ({len(lines)} lines total)"

        function_summaries.append({
            "function_name": t.name,
            "file_path": t.file_path,
            "cyclomatic_complexity": t.cyclomatic_complexity,
            "parameters": t.parameters,
            "lines": len(lines),
            "preview": preview,
        })

    user_msg = (
        f"## Optimization Targets ({len(targets)} functions)\n\n"
        f"Budget: optimize up to {max_functions} functions.\n\n"
        + json.dumps(function_summaries, indent=2)
    )

    _log(f"Using Claude Haiku for fast strategic planning...")

    try:
        plan = call_haiku_json(PLANNER_SYSTEM, user_msg)
    except Exception as e:
        _log(f"Planner failed ({e}), using default CC-descending order")
        plan = {
            "execution_order": [
                {
                    "function_name": t.name,
                    "file_path": t.file_path,
                    "priority": "high" if t.cyclomatic_complexity > 10 else "medium",
                    "reason": f"CC={t.cyclomatic_complexity}",
                    "suggested_strategy_hint": "",
                    "group": None,
                    "estimated_difficulty": "medium",
                }
                for t in targets[:max_functions]
            ],
            "groups": [],
            "estimated_quick_wins": 0,
            "planning_notes": "Default order by cyclomatic complexity",
        }

    # Log the plan
    execution_order = plan.get("execution_order", [])
    _log(f"Execution plan: {len(execution_order)} functions to optimize")

    for i, item in enumerate(execution_order[:max_functions]):
        priority = item.get("priority", "?")
        reason = item.get("reason", "")
        _log(f"  {i+1}. {item.get('function_name', '?')}() [{priority}] -- {reason}")

    groups = plan.get("groups", [])
    if groups:
        _log(f"Identified {len(groups)} function groups:")
        for g in groups:
            _log(f"  - {g.get('name', '?')}: {', '.join(g.get('members', []))}")

    quick_wins = plan.get("estimated_quick_wins", 0)
    if quick_wins:
        _log(f"Estimated quick wins: {quick_wins}")

    if emitter:
        emitter.complete(
            EventType.AGENT_LOG, "planner",
            f"Plan ready: {len(execution_order)} targets, {len(groups)} groups, {quick_wins} quick wins",
            data={
                "targets_planned": len(execution_order),
                "groups": len(groups),
                "quick_wins": quick_wins,
                "model": "haiku",
            }
        )

    return plan


def reorder_targets(
    targets: list[TargetFunction],
    plan: dict,
) -> list[TargetFunction]:
    """
    Reorder targets based on the planner's execution order.
    Falls back to original order if a target isn't in the plan.
    """
    execution_order = plan.get("execution_order", [])
    name_to_order = {
        item.get("function_name"): i
        for i, item in enumerate(execution_order)
    }

    def sort_key(t):
        if t.name in name_to_order:
            return name_to_order[t.name]
        return len(execution_order) + t.cyclomatic_complexity

    return sorted(targets, key=sort_key)


def get_strategy_hint(plan: dict, function_name: str) -> Optional[str]:
    """Get the planner's strategy hint for a specific function."""
    for item in plan.get("execution_order", []):
        if item.get("function_name") == function_name:
            return item.get("suggested_strategy_hint")
    return None


def get_plan_directives(plan: dict, function_name: str) -> Optional[dict]:
    """Get full plan directives for a specific function (for ConversationContext)."""
    for item in plan.get("execution_order", []):
        if item.get("function_name") == function_name:
            return {
                "priority": item.get("priority", "medium"),
                "strategy_hint": item.get("suggested_strategy_hint", ""),
                "estimated_difficulty": item.get("estimated_difficulty", "medium"),
                "group": item.get("group"),
                "reason": item.get("reason", ""),
            }
    return None
