"""
FastAPI Server — AGENTIC Pipeline Orchestrator.

Updated architecture with ConversationContext:
  1. Planner (Haiku) — Strategic optimization planning
  2. Triage (AST) — Function classification
  3. For each target (in planner's order):
     a. Create ConversationContext (shared working memory)
     b. Snapshot (deterministic) — Behavioral fingerprinting
     c. Multi-round optimization loop:
        - Analyst (Sonnet + tools) — AGENTIC hypothesis with tool use
        - Optimizer (Sonnet + tools) — SELF-VALIDATING code generation
        - Test Designer (Sonnet + tools) — SELF-VALIDATING test generation
        - Validator (deterministic) — Rubric pass/fail
        - If REJECTED -> Analyst diagnoses via ConversationContext -> loop
     d. All inter-agent messages visible via AGENT_HANDOFF events
  4. Results compiled + PR body generated

What makes this agentic:
  - Agents use Claude tool_use to autonomously call deterministic tools
  - Extended thinking traces visible in the agent feed
  - Multi-model: Haiku for planning, Sonnet for reasoning
  - Self-validating: Optimizer + Test Designer test their own output
  - Planner strategically orders optimization targets
  - ConversationContext: agents share working memory, see full history
  - AGENT_HANDOFF events: inter-agent communication visible in frontend
  - All events streamed via SSE in real time
"""

import asyncio
import uuid
import json
import time
import traceback
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.models.types import (
    TargetFunction, TriageResult, BehavioralSnapshot,
    AnalysisHypothesis, OptimizedCode, TestSuite,
    ValidationResult, Verdict, PipelineResult, RunResult,
    Category,
)
from backend.models.events import EventEmitter, EventType, PipelineEvent
from backend.core.conversation import ConversationContext

# Agent imports
from backend.agents.triage import triage_agent
from backend.agents.planner import planner_agent, reorder_targets, get_strategy_hint, get_plan_directives
from backend.agents.snapshot import snapshot_agent, lightweight_snapshot
from backend.agents.analyst import analyst_agent, diagnose_failure
from backend.agents.optimizer import optimizer_agent
from backend.agents.test_designer import test_designer_agent
from backend.agents.validator import validator_agent
from backend.agents.report_generator import report_generator_agent, format_report_as_pr_body


# ═══════════════════════════════════════════════════════════
# App Setup
# ═══════════════════════════════════════════════════════════

app = FastAPI(
    title="ComplexityImprover",
    description="Agentic Code Optimizer — HackEurope 2026",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory stores
runs: dict[str, RunResult] = {}
event_queues: dict[str, asyncio.Queue] = {}

MAX_ROUNDS = 3
MAX_FUNCTIONS = 10


# ═══════════════════════════════════════════════════════════
# Request/Response Models
# ═══════════════════════════════════════════════════════════

class OptimizeRequest(BaseModel):
    repo_path: str
    max_functions: int = MAX_FUNCTIONS
    skip_benchmarks: bool = False
    enable_thinking: bool = True    # Extended thinking for Analyst/Optimizer
    enable_planner: bool = True     # Strategic planning with Haiku


class OptimizeResponse(BaseModel):
    run_id: str
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str
    features: list[str]


# ═══════════════════════════════════════════════════════════
# Per-function optimization with ConversationContext
# ═══════════════════════════════════════════════════════════

async def run_optimization_for_function(
    ctx: ConversationContext,
    snapshot: BehavioralSnapshot,
    emitter: EventEmitter,
    skip_benchmarks: bool = False,
    enable_thinking: bool = True,
) -> tuple[PipelineResult, ConversationContext]:
    """
    The core multi-round conversation loop for a single function.

    Uses ConversationContext as shared working memory. Each agent reads
    what it needs from ctx, does its work, and writes output back.
    All inter-agent messages are visible via AGENT_HANDOFF events.
    """
    fname = ctx.function_name
    func_result = PipelineResult(
        function_name=fname,
        file_path=ctx.file_path,
    )

    hypothesis = None
    validation = None

    for round_num in range(1, ctx.max_rounds + 1):
        ctx.current_round = round_num
        emitter.complete(
            EventType.ROUND_START, "orchestrator",
            f"Round {round_num}/{ctx.max_rounds} for {fname}()",
            function_name=fname, round_number=round_num,
        )

        # -- ANALYST: form hypothesis (Round 1) or diagnose rejection (Round 2+) --
        if round_num == 1:
            hypothesis = await asyncio.to_thread(
                analyst_agent, snapshot, emitter, enable_thinking, ctx
            )
        else:
            hypothesis = await asyncio.to_thread(
                diagnose_failure, snapshot, validation,
                hypothesis, emitter, enable_thinking, ctx
            )

        # -- OPTIMIZER: implement hypothesis with self-validation --
        optimized = await asyncio.to_thread(
            optimizer_agent, snapshot, hypothesis,
            emitter, enable_thinking, ctx
        )

        if optimized is None:
            emitter.error(
                "orchestrator",
                f"Optimizer failed for {fname}() -- Round {round_num}",
                function_name=fname, round_number=round_num,
            )
            func_result.status = "error"
            break

        # -- TEST DESIGNER: generate diff-aware tests --
        test_suite = await asyncio.to_thread(
            test_designer_agent, snapshot, optimized,
            emitter, enable_thinking, ctx
        )

        # -- VALIDATOR: deterministic pass/fail --
        validation = await asyncio.to_thread(
            validator_agent, snapshot, optimized,
            test_suite, emitter, skip_benchmarks,
            None, ctx  # existing_test_command=None
        )

        func_result.validation = validation
        func_result.hypothesis = hypothesis
        func_result.rounds_taken = round_num

        if validation.verdict == Verdict.APPROVED:
            func_result.optimized_code = optimized
            func_result.status = "approved"
            emitter.log(
                "orchestrator",
                f"APPROVED: {fname}() in {round_num} round(s)!",
                function_name=fname, round_number=round_num,
            )
            break
        else:
            # Record rejection for next round's diagnosis
            vetoes = "; ".join(validation.veto_reasons[:2])
            remaining = ctx.max_rounds - round_num

            ctx.rejection_history.append({
                "round": round_num,
                "vetoes_triggered": validation.veto_reasons,
                "metric_details": {
                    m.metric_name: {"before": m.before, "after": m.after, "delta": m.delta_percent}
                    for m in validation.improvements
                },
                "summary": vetoes,
            })

            emitter.complete(
                EventType.REJECTION_DIAGNOSIS, "orchestrator",
                f"Round {round_num} rejected: {vetoes}. "
                f"{'Analyst will diagnose and adapt.' if remaining > 0 else 'Max rounds reached.'}",
                function_name=fname, round_number=round_num,
                data={
                    "round": round_num,
                    "vetoes": validation.veto_reasons,
                },
            )

            if round_num == ctx.max_rounds:
                func_result.status = "rejected"

    return func_result, ctx


# ═══════════════════════════════════════════════════════════
# THE AGENTIC PIPELINE
# ═══════════════════════════════════════════════════════════

async def run_pipeline(
    run_id: str,
    repo_path: str,
    max_functions: int = MAX_FUNCTIONS,
    skip_benchmarks: bool = False,
    enable_thinking: bool = True,
    enable_planner: bool = True,
):
    """
    Full agentic pipeline with:
    - Strategic planning (Haiku)
    - Autonomous tool use (Sonnet)
    - Self-validating code generation
    - ConversationContext per function
    - Multi-round feedback loop
    - Real-time SSE streaming
    """
    emitter = EventEmitter()
    queue = event_queues.get(run_id)

    if queue:
        emitter.on_event(lambda e: queue.put_nowait(e))

    result = RunResult(run_id=run_id, repo_path=repo_path)
    runs[run_id] = result

    try:
        # ═══════════════════════════════════════
        # PIPELINE START
        # ═══════════════════════════════════════
        emitter.complete(
            EventType.PIPELINE_START, "orchestrator",
            f"Starting agentic pipeline for {repo_path}",
            data={
                "features": {
                    "tool_use": True,
                    "extended_thinking": enable_thinking,
                    "multi_model": True,
                    "self_validation": True,
                    "strategic_planning": enable_planner,
                    "conversation_context": True,
                    "agent_handoffs": True,
                }
            }
        )

        # ═══════════════════════════════════════
        # STEP 1: TRIAGE (AST — fast)
        # ═══════════════════════════════════════
        emitter.log("triage", f"Scanning repository: {repo_path}")
        triage_dict = await asyncio.to_thread(triage_agent, repo_path)

        targets = []
        for t in triage_dict.get("targets", []):
            targets.append(TargetFunction(
                name=t["function_name"],
                file_path=t["file_path"],
                source_code=t["source_code"],
                start_line=t.get("start_line", 0),
                end_line=t.get("end_line", 0),
                cyclomatic_complexity=t.get("cyclomatic_complexity", 0),
                parameters=t.get("parameters", []),
                category=Category.A,
                reason=t.get("reason", ""),
            ))

        # Build Category B targets
        category_b_targets = []
        for b in triage_dict.get("analysis_only", []):
            category_b_targets.append(TargetFunction(
                name=b["function_name"],
                file_path=b["file_path"],
                source_code=b.get("source_code", ""),
                start_line=b.get("start_line", 0),
                end_line=b.get("end_line", 0),
                cyclomatic_complexity=b.get("cyclomatic_complexity", 0),
                parameters=b.get("parameters", []),
                category=Category.B,
                reason=b.get("reason", "Side effects detected"),
                red_flags=b.get("red_flags", []),
            ))

        # Build Category C targets
        category_c_targets = []
        for c in triage_dict.get("skipped", []):
            category_c_targets.append(TargetFunction(
                name=c.get("function_name", ""),
                file_path=c.get("file_path", ""),
                source_code=c.get("source_code", ""),
                start_line=c.get("start_line", 0),
                end_line=c.get("end_line", 0),
                cyclomatic_complexity=c.get("cyclomatic_complexity", 0),
                parameters=c.get("parameters", []),
                category=Category.C,
                reason=c.get("reason", "Skipped"),
            ))

        triage_result = TriageResult(
            category_a=targets,
            category_b=category_b_targets,
            category_c=category_c_targets,
            repo_path=repo_path,
            test_framework=triage_dict.get("test_framework"),
            test_dir=triage_dict.get("test_directory"),
        )
        result.triage = triage_result
        result.total_functions = len(targets)

        emitter.complete(
            EventType.TRIAGE_COMPLETE, "triage",
            f"Found {len(targets)} optimization targets",
            data={
                "category_a": len(targets),
                "category_b": len(triage_dict.get("analysis_only", [])),
                "category_c": len(triage_dict.get("skipped", [])),
            }
        )

        if not targets:
            emitter.complete(
                EventType.PIPELINE_COMPLETE, "orchestrator",
                "No optimization targets found",
            )
            return

        # ═══════════════════════════════════════
        # STEP 2: PLANNER (Haiku — fast + strategic)
        # ═══════════════════════════════════════
        plan = None
        if enable_planner and len(targets) > 1:
            emitter.log("planner", "Strategic planning with Claude Haiku...")
            plan = await asyncio.to_thread(
                planner_agent, targets, max_functions, emitter
            )
            targets = reorder_targets(targets, plan)
            emitter.complete(
                EventType.PLAN_COMPLETE, "planner",
                f"Optimization plan ready -- {len(targets)} targets ordered strategically",
                data={"model": "haiku"}
            )

        targets_to_process = targets[:max_functions]

        # ═══════════════════════════════════════
        # STEP 3-7: Process each function (AGENTIC with ConversationContext)
        # ═══════════════════════════════════════
        for i, target in enumerate(targets_to_process):
            fname = target.name

            emitter.log(
                "orchestrator",
                f"--- Processing {i+1}/{len(targets_to_process)}: {fname}() ---",
                function_name=fname
            )

            try:
                # -- Create ConversationContext for this function --
                ctx = ConversationContext(
                    function_name=fname,
                    original_source=target.source_code,
                    file_path=target.file_path,
                    category="A",
                    max_rounds=MAX_ROUNDS,
                )
                ctx.set_emitter(emitter)

                # -- Inject planner directives --
                if plan:
                    directives = get_plan_directives(plan, fname)
                    if directives:
                        ctx.plan_directives = directives
                        ctx.add_message(
                            sender="planner",
                            recipient="all",
                            message_type="plan_directive",
                            content=directives,
                        )

                    strategy_hint = get_strategy_hint(plan, fname)
                    if strategy_hint:
                        emitter.log("planner",
                                   f"Hint for {fname}(): {strategy_hint}",
                                   function_name=fname)

                # -- SNAPSHOT (deterministic) --
                snapshot = await asyncio.to_thread(
                    snapshot_agent, target, emitter, skip_benchmarks
                )
                ctx.snapshot_metrics = {
                    "cyclomatic_complexity": snapshot.static_metrics.cyclomatic_complexity,
                    "maintainability_index": snapshot.static_metrics.maintainability_index,
                    "big_o": snapshot.big_o_estimate,
                    "big_o_slope": snapshot.big_o_slope,
                }

                # -- MULTI-ROUND OPTIMIZATION LOOP (via ConversationContext) --
                func_result, ctx = await run_optimization_for_function(
                    ctx=ctx,
                    snapshot=snapshot,
                    emitter=emitter,
                    skip_benchmarks=skip_benchmarks,
                    enable_thinking=enable_thinking,
                )

                if func_result.status == "approved":
                    result.optimized_count += 1
                elif func_result.status == "rejected":
                    result.rejected_count += 1

            except Exception as e:
                func_result = PipelineResult(
                    function_name=fname,
                    file_path=target.file_path,
                    status="error",
                )
                result.error_count += 1
                emitter.error(
                    "orchestrator",
                    f"Error processing {fname}(): {str(e)}",
                    function_name=fname,
                )
                traceback.print_exc()

            result.results.append(func_result)

        # ═══════════════════════════════════════
        # CATEGORY B: Analysis-only (metrics, no execution)
        # ═══════════════════════════════════════
        if category_b_targets:
            emitter.log(
                "orchestrator",
                f"Analyzing {len(category_b_targets)} Category B functions (metrics only)...",
            )

            for b_target in category_b_targets:
                try:
                    metrics = await asyncio.to_thread(
                        lightweight_snapshot, b_target, emitter
                    )
                    result.category_b_reports.append({
                        "function_name": b_target.name,
                        "file_path": b_target.file_path,
                        "category": "B",
                        "reason": b_target.reason,
                        "red_flags": b_target.red_flags,
                        "metrics": metrics,
                        "recommendation": "Manual optimization recommended -- contains side effects",
                    })
                except Exception as e:
                    emitter.error(
                        "orchestrator",
                        f"Category B analysis failed for {b_target.name}(): {e}",
                    )

            emitter.log(
                "orchestrator",
                f"Category B analysis complete: {len(result.category_b_reports)} reports",
            )

        # ═══════════════════════════════════════
        # REPORT GENERATOR
        # ═══════════════════════════════════════
        emitter.log("orchestrator", "Generating final report...")
        report = await asyncio.to_thread(
            report_generator_agent, result, emitter
        )

        # ═══════════════════════════════════════
        # PIPELINE COMPLETE
        # ═══════════════════════════════════════
        emitter.complete(
            EventType.PIPELINE_COMPLETE, "orchestrator",
            f"Pipeline complete: {result.optimized_count} optimized, "
            f"{result.rejected_count} rejected, {result.error_count} errors "
            f"out of {len(targets_to_process)} targets",
            data={
                "optimized": result.optimized_count,
                "rejected": result.rejected_count,
                "errors": result.error_count,
                "total": len(targets_to_process),
                "category_b_analyzed": len(result.category_b_reports),
                "report": report.get("summary", {}),
                "features_used": {
                    "tool_use": True,
                    "extended_thinking": enable_thinking,
                    "multi_model": enable_planner,
                    "self_validation": True,
                    "conversation_context": True,
                    "agent_handoffs": True,
                    "rounds_used": sum(r.rounds_taken for r in result.results),
                }
            }
        )

    except Exception as e:
        emitter.error("orchestrator", f"Pipeline failed: {str(e)}")
        traceback.print_exc()

    finally:
        if queue:
            await queue.put(None)  # Signal stream end


# ═══════════════════════════════════════════════════════════
# API Endpoints
# ═══════════════════════════════════════════════════════════

@app.get("/api/health")
async def health():
    return HealthResponse(
        status="ok",
        version="0.3.0",
        features=[
            "tool_use", "extended_thinking", "multi_model",
            "self_validation", "strategic_planning", "sse_streaming",
            "conversation_context", "agent_handoffs",
            "web_research", "pattern_database",
        ]
    )


@app.post("/api/optimize")
async def start_optimization(
    request: OptimizeRequest,
    background_tasks: BackgroundTasks,
):
    """Start the agentic optimization pipeline."""
    run_id = str(uuid.uuid4())
    event_queues[run_id] = asyncio.Queue()

    background_tasks.add_task(
        run_pipeline,
        run_id=run_id,
        repo_path=request.repo_path,
        max_functions=request.max_functions,
        skip_benchmarks=request.skip_benchmarks,
        enable_thinking=request.enable_thinking,
        enable_planner=request.enable_planner,
    )

    return OptimizeResponse(run_id=run_id, message="Agentic pipeline started")


@app.get("/api/stream/{run_id}")
async def stream_events(run_id: str):
    """SSE endpoint -- streams ALL pipeline events in real time."""
    queue = event_queues.get(run_id)
    if not queue:
        raise HTTPException(404, f"Run {run_id} not found")

    async def event_generator():
        while True:
            event = await queue.get()
            if event is None:
                yield "data: {\"event_type\": \"STREAM_END\"}\n\n"
                break
            if isinstance(event, PipelineEvent):
                yield event.to_sse()
            else:
                yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/results/{run_id}")
async def get_results(run_id: str):
    """Get completed pipeline results."""
    result = runs.get(run_id)
    if not result:
        raise HTTPException(404, f"Run {run_id} not found")
    return _serialize_run_result(result)


@app.get("/api/events/{run_id}")
async def get_all_events(run_id: str):
    """Get all events for a completed run (for replay)."""
    result = runs.get(run_id)
    if not result:
        raise HTTPException(404, f"Run {run_id} not found")

    return {"run_id": run_id, "message": "Use /api/stream/{run_id} for real-time events"}


# ═══════════════════════════════════════════════════════════
# Serialization
# ═══════════════════════════════════════════════════════════

def _serialize_run_result(result: RunResult) -> dict:
    out = {
        "run_id": result.run_id,
        "repo_path": result.repo_path,
        "total_functions": result.total_functions,
        "optimized_count": result.optimized_count,
        "rejected_count": result.rejected_count,
        "error_count": result.error_count,
        "results": [],
        "category_b_reports": result.category_b_reports,
        "category_c_skipped": [
            {
                "function_name": t.name,
                "file_path": t.file_path,
                "reason": t.reason,
            }
            for t in (result.triage.category_c if result.triage else [])
        ],
    }

    for pr in result.results:
        entry = {
            "function_name": pr.function_name,
            "file_path": pr.file_path,
            "status": pr.status,
            "rounds_taken": pr.rounds_taken,
        }

        if pr.optimized_code:
            entry["optimized_code"] = {
                "original_source": pr.optimized_code.original_source,
                "optimized_source": pr.optimized_code.optimized_source,
                "changes_description": pr.optimized_code.changes_description,
                "strategy_used": pr.optimized_code.strategy_used,
            }

        if pr.validation:
            v = pr.validation
            entry["validation"] = {
                "verdict": v.verdict.value,
                "veto_reasons": v.veto_reasons,
                "improvements": [
                    {
                        "metric": m.metric_name,
                        "before": m.before,
                        "after": m.after,
                        "improved": m.improved,
                        "delta_percent": m.delta_percent,
                    }
                    for m in v.improvements
                ],
                "tests": {
                    "passed": v.test_results.passed if v.test_results else 0,
                    "failed": v.test_results.failed if v.test_results else 0,
                } if v.test_results else None,
            }

        if pr.hypothesis:
            h = pr.hypothesis
            entry["hypothesis"] = {
                "strategy": h.strategy,
                "current_complexity": h.current_complexity,
                "proposed_complexity": h.proposed_complexity,
                "bottleneck": h.bottleneck,
            }

        out["results"].append(entry)

    return out


# ═══════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
