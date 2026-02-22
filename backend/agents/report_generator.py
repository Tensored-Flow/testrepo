"""
Agent 7: Report Generator — Compiles final optimization report.

Block 15 from the Miro workflow. Takes the full pipeline results and generates:
  - Before/after metrics table
  - Performance comparison data (for frontend charts)
  - Agent reasoning summary
  - Test coverage summary
  - Code diff summary with confidence score
  - Category B analysis-only reports
  - Optional LLM-generated executive summary (Haiku — fast + cheap)
"""

import json
from typing import Optional

from backend.models.types import (
    RunResult, PipelineResult, ValidationResult, Verdict,
    BehavioralSnapshot, MetricsComparison,
)
from backend.models.events import EventEmitter, EventType
from backend.integrations.llm_client import call_haiku


# ═══════════════════════════════════════════════════════════
# Confidence Scoring
# ═══════════════════════════════════════════════════════════

def _compute_confidence(result: PipelineResult) -> int:
    """
    Compute a confidence score (0-100) for an optimization.
    Based on: tests passed, metric improvements, rounds taken.
    """
    score = 0

    if result.status != "approved":
        return 0

    v = result.validation
    if not v:
        return 0

    # Tests passing = high confidence
    if v.test_results:
        if v.test_results.failed == 0:
            score += 40  # All tests pass
        total = v.test_results.passed + v.test_results.failed
        if total > 5:
            score += 10  # Good test coverage

    # Number of improved metrics
    improved_count = sum(1 for m in v.improvements if m.improved)
    score += min(improved_count * 5, 25)  # Up to 25 points for metric improvements

    # Fewer rounds = higher confidence (got it right quickly)
    if result.rounds_taken == 1:
        score += 15
    elif result.rounds_taken == 2:
        score += 10
    else:
        score += 5

    # No veto reasons at all
    if not v.veto_reasons:
        score += 10

    return min(score, 100)


# ═══════════════════════════════════════════════════════════
# Metrics Extraction
# ═══════════════════════════════════════════════════════════

def _extract_metrics(snapshot: Optional[BehavioralSnapshot]) -> dict:
    """Extract a clean metrics dict from a BehavioralSnapshot."""
    if not snapshot:
        return {}

    sm = snapshot.static_metrics
    return {
        "cyclomatic_complexity": sm.cyclomatic_complexity,
        "cognitive_complexity": sm.cognitive_complexity,
        "maintainability_index": round(sm.maintainability_index, 2),
        "halstead_volume": round(sm.halstead_volume, 2),
        "halstead_difficulty": round(sm.halstead_difficulty, 2),
        "halstead_bugs": round(sm.halstead_bugs, 4),
        "loc": sm.loc,
        "nesting_depth": sm.nesting_depth,
        "big_o": snapshot.big_o_estimate,
        "big_o_slope": round(snapshot.big_o_slope, 3),
        "bytecode_instructions": snapshot.bytecode_instruction_count,
    }


def _build_improvements_table(validation: ValidationResult) -> list[dict]:
    """Build a clean improvements table from validation metrics."""
    return [
        {
            "metric": m.metric_name,
            "before": round(m.before, 4),
            "after": round(m.after, 4),
            "delta_percent": round(m.delta_percent, 2),
            "improved": m.improved,
        }
        for m in validation.improvements
    ]


# ═══════════════════════════════════════════════════════════
# Executive Summary (Haiku — fast + cheap)
# ═══════════════════════════════════════════════════════════

def _generate_executive_summary(report: dict) -> str:
    """Use Haiku to generate a 2-3 sentence executive summary."""
    summary = report.get("summary", {})
    optimizations = report.get("optimizations", [])

    # Build a quick stats string for Haiku
    stats = (
        f"Analyzed {summary.get('total_targets', 0)} functions. "
        f"Successfully optimized {summary.get('optimized', 0)}, "
        f"rejected {summary.get('rejected', 0)}, "
        f"errors {summary.get('errors', 0)}."
    )

    if optimizations:
        strategies = [o.get("strategy", "") for o in optimizations[:3]]
        improvements = []
        for o in optimizations:
            for imp in o.get("improvements", []):
                if imp.get("improved"):
                    improvements.append(
                        f"{imp['metric']}: {imp['delta_percent']:+.1f}%"
                    )
        stats += f" Key strategies: {'; '.join(strategies[:3])}."
        if improvements:
            stats += f" Notable improvements: {', '.join(improvements[:5])}."

    try:
        summary_text = call_haiku(
            "Write a 2-3 sentence executive summary of this code optimization run. "
            "Be specific about what improved. Professional tone.",
            stats,
            max_tokens=200,
        )
        return summary_text.strip()
    except Exception:
        return (
            f"Optimized {summary.get('optimized', 0)} of {summary.get('total_targets', 0)} "
            f"target functions with measurable improvements in complexity and performance metrics."
        )


# ═══════════════════════════════════════════════════════════
# Main Agent Entry Point
# ═══════════════════════════════════════════════════════════

def report_generator_agent(
    run_result: RunResult,
    emitter: Optional[EventEmitter] = None,
    include_executive_summary: bool = True,
) -> dict:
    """
    Compile the final optimization report.

    Args:
        run_result: The completed pipeline RunResult
        emitter: Optional event emitter for SSE
        include_executive_summary: Whether to call Haiku for a summary

    Returns:
        Complete report dict ready for frontend rendering + PR body
    """
    _log = lambda msg: emitter.log("report", msg) if emitter else None

    _log("📊 Compiling optimization report...")

    # ── Count events for stats ──
    total_tool_calls = 0
    total_thinking_traces = 0
    if emitter:
        for e in emitter.events:
            if e.event_type == EventType.TOOL_CALL:
                total_tool_calls += 1
            elif e.event_type == EventType.THINKING:
                total_thinking_traces += 1

    # ── Summary stats ──
    report = {
        "summary": {
            "total_targets": run_result.total_functions,
            "optimized": run_result.optimized_count,
            "rejected": run_result.rejected_count,
            "errors": run_result.error_count,
            "total_rounds_used": sum(r.rounds_taken for r in run_result.results),
            "total_tool_calls": total_tool_calls,
            "total_thinking_traces": total_thinking_traces,
        },
        "optimizations": [],
        "rejected": [],
        "category_b_reports": getattr(run_result, 'category_b_reports', []),
        "category_c_skipped": [],
        "executive_summary": "",
    }

    # ── Category C from triage ──
    if run_result.triage:
        report["category_c_skipped"] = [
            {
                "function_name": t.name,
                "file_path": t.file_path,
                "reason": t.reason,
            }
            for t in run_result.triage.category_c
        ]

    # ── Process each function result ──
    for pr in run_result.results:
        if pr.status == "approved" and pr.validation and pr.optimized_code:
            v = pr.validation
            confidence = _compute_confidence(pr)

            entry = {
                "function_name": pr.function_name,
                "file_path": pr.file_path,
                "status": "approved",
                "rounds": pr.rounds_taken,
                "strategy": pr.hypothesis.strategy if pr.hypothesis else "",
                "bottleneck": pr.hypothesis.bottleneck if pr.hypothesis else "",
                "complexity_change": (
                    f"{pr.hypothesis.current_complexity} → {pr.hypothesis.proposed_complexity}"
                    if pr.hypothesis else ""
                ),
                "before_metrics": _extract_metrics(v.before_snapshot),
                "after_metrics": _extract_metrics(v.after_snapshot),
                "improvements": _build_improvements_table(v),
                "confidence_score": confidence,
                "code_diff": {
                    "original_loc": len(pr.optimized_code.original_source.split("\n")),
                    "optimized_loc": len(pr.optimized_code.optimized_source.split("\n")),
                    "changes_description": pr.optimized_code.changes_description,
                },
                "tests": {
                    "passed": v.test_results.passed if v.test_results else 0,
                    "failed": v.test_results.failed if v.test_results else 0,
                },
            }
            report["optimizations"].append(entry)
            _log(f"  ✅ {pr.function_name}(): confidence={confidence}%, {pr.rounds_taken} round(s)")

        elif pr.status == "rejected":
            entry = {
                "function_name": pr.function_name,
                "file_path": pr.file_path,
                "status": "rejected",
                "rounds": pr.rounds_taken,
                "best_strategy": pr.hypothesis.strategy if pr.hypothesis else "No strategy formed",
                "veto_reasons": pr.validation.veto_reasons if pr.validation else [],
            }
            report["rejected"].append(entry)
            _log(f"  ❌ {pr.function_name}(): rejected after {pr.rounds_taken} round(s)")

        elif pr.status == "error":
            report["rejected"].append({
                "function_name": pr.function_name,
                "file_path": pr.file_path,
                "status": "error",
                "rounds": pr.rounds_taken,
                "best_strategy": "",
                "veto_reasons": ["Pipeline error"],
            })
            _log(f"  ⚠️ {pr.function_name}(): error")

    # ── Executive summary (Haiku) ──
    if include_executive_summary:
        _log("✍️ Generating executive summary with Claude Haiku...")
        report["executive_summary"] = _generate_executive_summary(report)
        _log(f"📝 {report['executive_summary']}")

    _log(
        f"📊 Report complete: {report['summary']['optimized']} optimized, "
        f"{report['summary']['rejected']} rejected, "
        f"confidence avg={_avg_confidence(report)}%"
    )

    if emitter:
        emitter.complete(
            EventType.AGENT_LOG, "report",
            "Report generation complete",
            data={
                "optimized": report["summary"]["optimized"],
                "confidence_avg": _avg_confidence(report),
                "model": "haiku",
            }
        )

    return report


def _avg_confidence(report: dict) -> int:
    """Average confidence score across optimizations."""
    scores = [o.get("confidence_score", 0) for o in report.get("optimizations", [])]
    return round(sum(scores) / len(scores)) if scores else 0


# ═══════════════════════════════════════════════════════════
# Format for GitHub PR
# ═══════════════════════════════════════════════════════════

def format_report_as_pr_body(report: dict) -> str:
    """Convert the report into a markdown GitHub PR body."""
    lines = [
        "## 🚀 Code Optimization by ComplexityImprover",
        "",
        f"*{report.get('executive_summary', '')}*",
        "",
        f"**Summary:** {report['summary']['optimized']} functions optimized out of "
        f"{report['summary']['total_targets']} targets "
        f"({report['summary']['total_rounds_used']} total optimization rounds, "
        f"{report['summary']['total_tool_calls']} autonomous tool calls)",
        "",
    ]

    if report["optimizations"]:
        lines.append("### ✅ Optimizations")
        lines.append("")

        for o in report["optimizations"]:
            lines.append(f"#### `{o['function_name']}()` in `{o['file_path']}`")
            lines.append(f"**Strategy:** {o['strategy']}")
            if o.get("complexity_change"):
                lines.append(f"**Complexity:** {o['complexity_change']}")
            lines.append(f"**Confidence:** {o['confidence_score']}%")

            improved = [i for i in o.get("improvements", []) if i.get("improved")]
            if improved:
                lines.append("**Improvements:**")
                for imp in improved:
                    lines.append(f"- {imp['metric']}: {imp['before']} → {imp['after']} ({imp['delta_percent']:+.1f}%)")

            lines.append(f"**Tests:** {o['tests']['passed']} passed, {o['tests']['failed']} failed")
            lines.append("")

    if report["rejected"]:
        lines.append("### ❌ Could Not Optimize")
        lines.append("")
        for r in report["rejected"]:
            lines.append(f"- `{r['function_name']}()`: {'; '.join(r.get('veto_reasons', ['Unknown']))}")
        lines.append("")

    lines.extend([
        "---",
        "*Generated by ComplexityImprover — AI proposes, math disposes. "
        "Every improvement proven with deterministic metrics.*",
    ])

    return "\n".join(lines)