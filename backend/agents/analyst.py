"""
Agent 3: Analyst — AGENTIC Hypothesis Formation.

THIS IS THE SHOWCASE AGENT. Claude doesn't just receive pre-computed metrics —
it DECIDES which metrics to compute, calls the tools autonomously, interprets
the results, and forms an optimization hypothesis.

On Round 2+, it diagnoses WHY the previous optimization was rejected
by autonomously investigating with tools.

Features that win Agentic AI + Best Use of Claude:
  - Claude tool_use API — agent autonomously calls deterministic tools
  - Extended thinking — visible reasoning traces in agent feed
  - Multi-turn tool loop — Claude calls tools, reads results, calls more tools
  - Failure diagnosis — agents CONVERSE about what went wrong
  - ConversationContext — full inter-agent conversation history
  - Web research — can look up algorithms and best practices
  - Pattern database — checks known antipatterns before heavy analysis
"""

import json
from typing import Optional

from backend.models.types import (
    BehavioralSnapshot, AnalysisHypothesis, ValidationResult,
)
from backend.models.events import EventEmitter, EventType
from backend.integrations.llm_client import call_claude_with_tools, SONNET
from backend.core.tools import ANALYST_TOOLS, execute_tool
from backend.core.conversation import ConversationContext


# ═══════════════════════════════════════════════════════════
# System Prompts — Now round-aware with conversation context
# ═══════════════════════════════════════════════════════════

def build_analyst_system_prompt(ctx: Optional[ConversationContext] = None) -> str:
    """Build the analyst system prompt, incorporating conversation context for multi-round awareness."""
    base = """\
You are an expert performance engineer analyzing a Python function for optimization.

You have access to TOOLS that let you measure the function precisely:
- run_lizard: Compute cyclomatic complexity
- run_radon: Compute Halstead metrics + Maintainability Index
- analyze_bytecode: Disassemble to CPython bytecode
- benchmark_function: Time the function at multiple input sizes
- estimate_big_o: Estimate Big O from benchmark data
- run_function_with_inputs: Test the function with specific inputs
- web_research: Search the web for algorithm optimization strategies
- lookup_optimization_pattern: Check for known antipatterns with standard fixes
- analyze_code_diff: Compare original vs optimized code (Round 2+)
- compare_metrics_detailed: Side-by-side metric comparison (Round 2+)
- generate_failure_hypotheses: Structured failure diagnosis (Round 2+)

YOUR INVESTIGATION PROCESS:
1. FIRST: Use lookup_optimization_pattern to check for known antipatterns
2. THEN: Use run_lizard and run_radon for complexity metrics
3. IF NEEDED: Use analyze_bytecode for low-level insights
4. IF UNFAMILIAR: Use web_research to look up libraries, algorithms, or best practices
5. FINALLY: Use benchmark_function and estimate_big_o for empirical data

For example:
- If you see nested loops → call run_lizard to confirm complexity, then benchmark to measure actual growth rate
- If you see many function calls → call analyze_bytecode to see call overhead
- If you're unsure what the function does → call run_function_with_inputs with test data

After gathering evidence, provide your optimization hypothesis as your final text response in this EXACT JSON format:
{
  "current_complexity": "O(...)",
  "proposed_complexity": "O(...)",
  "bottleneck": "specific problem identified from your measurements",
  "strategy": "concrete optimization approach backed by your evidence",
  "expected_speedup": "realistic estimate based on benchmarks",
  "risks": ["what could go wrong"]
}

Rules:
- ALWAYS call at least 2 tools before forming a hypothesis — you need evidence
- Be SPECIFIC: cite actual numbers from your measurements
- The strategy must be implementable: "Replace the O(n) list search on line 7 with a dict lookup"
- Don't suggest changes that alter function behavior
- Include in your hypothesis: exact lines to change, the specific technique, expected complexity change, and confidence level"""

    if ctx and ctx.current_round > 1:
        base += f"""

=== CRITICAL: THIS IS ROUND {ctx.current_round} — PREVIOUS ATTEMPT(S) FAILED ===

{ctx.get_rejection_summary()}

You MUST:
1. Use analyze_code_diff to understand what the Optimizer actually changed last round
2. Use compare_metrics_detailed to see exactly which metrics regressed
3. Use generate_failure_hypotheses to structure your diagnosis
4. Form a DIFFERENT hypothesis that avoids the same failure mode
5. If the same approach keeps failing, try a fundamentally different optimization strategy
6. Use web_research if you need to find alternative approaches

DO NOT repeat the same hypothesis. The definition of insanity is trying the same thing and expecting different results.

Include a "failure_diagnosis" field in your JSON response explaining what went wrong."""

    if ctx:
        briefing = ctx.to_agent_briefing("analyst")
        if briefing and ctx.messages:
            base += f"""

=== CONVERSATION HISTORY ===
{briefing}"""

        if ctx.plan_directives:
            base += f"""

=== PLANNER DIRECTIVES ===
{ctx.plan_directives}"""

    return base


DIAGNOSIS_SYSTEM = """\
You are an expert performance engineer diagnosing why a code optimization was REJECTED.

You have measurement tools AND diagnostic tools available. The previous optimization attempt FAILED.
Your job: investigate WHY it failed and propose a DIFFERENT strategy.

You have tools to investigate:
- run_function_with_inputs: Test the failed optimization with specific inputs to find behavioral differences
- compare_outputs (if available): Directly compare original vs optimized outputs
- run_lizard / run_radon: Check if the "optimization" actually worsened metrics
- benchmark_function: Check if it's actually slower
- analyze_code_diff: See exactly what changed between original and optimized
- compare_metrics_detailed: Side-by-side metric comparison
- generate_failure_hypotheses: Get structured diagnosis based on vetoes
- web_research: Research alternative optimization approaches
- lookup_optimization_pattern: Check for known patterns to try instead

INVESTIGATION PROCESS:
1. Use generate_failure_hypotheses to understand the rejection
2. Use analyze_code_diff to see what actually changed
3. Use compare_metrics_detailed to see metric impact
4. Form a NEW hypothesis that avoids the same failure

After investigating, respond with JSON:
{
  "current_complexity": "O(...)",
  "proposed_complexity": "O(...)",
  "bottleneck": "...",
  "strategy": "DIFFERENT strategy that avoids the failure",
  "expected_speedup": "...",
  "risks": ["..."],
  "failure_diagnosis": "what specifically went wrong, backed by your tool measurements"
}

CRITICAL: Your new strategy MUST be fundamentally different from the one that failed.
"""


# ═══════════════════════════════════════════════════════════
# Event-emitting tool executor (wraps the raw executor)
# ═══════════════════════════════════════════════════════════

def _make_event_tool_executor(emitter: Optional[EventEmitter], fname: str, round_num: int):
    """
    Wraps execute_tool to emit events for every tool call.
    This is what makes the agent feed show Claude's investigation in real time.
    """
    def executor(tool_name: str, tool_input: dict) -> str:
        # Emit: Claude is calling a tool
        if emitter:
            input_summary = _summarize_tool_input(tool_name, tool_input)
            emitter.log(
                "analyst", f"Calling {tool_name}({input_summary})",
                function_name=fname, round_number=round_num,
                data={"tool": tool_name, "action": "call"}
            )

        # Execute deterministically
        result_str = execute_tool(tool_name, tool_input)

        # Emit: tool result summary
        if emitter:
            result_summary = _summarize_tool_result(tool_name, result_str)
            emitter.log(
                "analyst", f"{tool_name} -> {result_summary}",
                function_name=fname, round_number=round_num,
                data={"tool": tool_name, "action": "result", "summary": result_summary}
            )

        return result_str

    return executor


def _summarize_tool_input(tool_name: str, tool_input: dict) -> str:
    """Human-readable summary of tool input for the agent feed."""
    if tool_name == "benchmark_function":
        sizes = tool_input.get("sizes", [100, 500, 1000, 5000])
        return f"sizes={sizes}"
    elif tool_name == "run_function_with_inputs":
        n = len(tool_input.get("test_inputs", []))
        return f"{n} test inputs"
    elif tool_name == "analyze_bytecode":
        return tool_input.get("function_name", "")
    elif tool_name == "estimate_big_o":
        return f"{len(tool_input.get('sizes', []))} data points"
    elif tool_name == "web_research":
        return f"'{tool_input.get('query', '')[:50]}'"
    elif tool_name == "lookup_optimization_pattern":
        return f"'{tool_input.get('pattern_description', '')[:50]}'"
    elif tool_name == "analyze_code_diff":
        return "original vs optimized"
    elif tool_name == "compare_metrics_detailed":
        return tool_input.get("function_name", "")
    elif tool_name == "generate_failure_hypotheses":
        vetoes = tool_input.get("vetoes_triggered", [])
        return f"{len(vetoes)} vetoes"
    return ""


def _summarize_tool_result(tool_name: str, result_str: str) -> str:
    """Human-readable summary of tool result for the agent feed."""
    try:
        data = json.loads(result_str)
    except (json.JSONDecodeError, TypeError):
        return result_str[:100]

    if tool_name == "run_lizard":
        cc = data.get("cyclomatic_complexity", "?")
        return f"CC={cc}"
    elif tool_name == "run_radon":
        mi = data.get("maintainability_index", "?")
        hd = data.get("halstead_difficulty", "?")
        return f"MI={mi}, Halstead difficulty={hd}"
    elif tool_name == "analyze_bytecode":
        total = data.get("total_instructions", "?")
        cats = data.get("categories", {})
        top = max(cats.items(), key=lambda x: x[1])[0] if cats else "?"
        return f"{total} instructions, heaviest: {top}"
    elif tool_name == "benchmark_function":
        benchmarks = data.get("benchmarks", [])
        if benchmarks and isinstance(benchmarks[-1], dict):
            t = benchmarks[-1].get("mean_time", 0)
            s = benchmarks[-1].get("size", "?")
            return f"n={s}: {t*1000:.2f}ms"
        return f"{len(benchmarks)} sizes benchmarked"
    elif tool_name == "estimate_big_o":
        return f"{data.get('big_o', '?')} (slope={data.get('slope', '?')}, p={data.get('p_value', '?')})"
    elif tool_name == "run_function_with_inputs":
        n = data.get("total_tests", "?")
        return f"{n} inputs tested"
    elif tool_name == "compare_outputs":
        if data.get("all_match"):
            return f"All {data.get('total_tests', '?')} outputs match"
        else:
            return f"{data.get('mismatches', '?')} mismatches found"
    elif tool_name == "web_research":
        findings = data.get("findings", "")
        return findings[:100] + "..." if len(findings) > 100 else findings
    elif tool_name == "lookup_optimization_pattern":
        n = data.get("matches_found", 0)
        return f"{n} pattern(s) matched"
    elif tool_name == "analyze_code_diff":
        return f"{data.get('additions', 0)} additions, {data.get('deletions', 0)} deletions"
    elif tool_name == "compare_metrics_detailed":
        improved = sum(1 for c in data.get("comparisons", []) if c.get("improved"))
        return f"{improved}/{len(data.get('comparisons', []))} metrics improved"
    elif tool_name == "generate_failure_hypotheses":
        return f"{data.get('total', 0)} hypotheses generated"

    return json.dumps(data)[:150]


# ═══════════════════════════════════════════════════════════
# Main Agent Entry Points
# ═══════════════════════════════════════════════════════════

def analyst_agent(
    snapshot: BehavioralSnapshot,
    emitter: Optional[EventEmitter] = None,
    enable_thinking: bool = True,
    ctx: Optional[ConversationContext] = None,
) -> AnalysisHypothesis:
    """
    Round 1: AGENTIC analysis with tool use.

    Claude autonomously investigates the function by calling tools,
    interpreting results, and forming an optimization hypothesis.

    Args:
        snapshot: Behavioral snapshot of the function
        emitter: Event emitter for SSE streaming
        enable_thinking: Enable extended thinking
        ctx: Optional ConversationContext for multi-agent communication
    """
    fname = snapshot.function_name
    _log = lambda msg: emitter.log("analyst", msg, function_name=fname) if emitter else None

    _log(f"Analyst investigating {fname}() -- autonomous tool use enabled")

    # Build the user message with the source code
    user_msg = (
        f"Analyze this Python function and form an optimization hypothesis.\n\n"
        f"## Function: {fname}()\n"
        f"**File:** {snapshot.file_path}\n"
        f"**Parameters:** {', '.join(snapshot.static_metrics.parameter_count * ['param']) or 'none'}\n\n"
        f"```python\n{snapshot.source_code}\n```\n\n"
        f"Use your tools to measure this function's complexity, performance, and behavior. "
        f"Then form a specific, evidence-backed optimization hypothesis."
    )

    # Build round-aware system prompt
    system_prompt = build_analyst_system_prompt(ctx)

    # Create event-emitting tool executor
    executor = _make_event_tool_executor(emitter, fname, round_num=1)

    # Callbacks for thinking traces
    def on_thinking(text):
        if emitter:
            preview = text[:200] + "..." if len(text) > 200 else text
            emitter.log("analyst", f"[thinking] {preview}",
                       function_name=fname, data={"type": "thinking"})

    # THE AGENTIC CALL — Claude decides which tools to use
    result = call_claude_with_tools(
        system_prompt=system_prompt,
        user_message=user_msg,
        tools=ANALYST_TOOLS,
        tool_executor=executor,
        enable_thinking=enable_thinking,
        on_thinking=on_thinking,
        max_turns=10,
    )

    # Parse the hypothesis from Claude's final response
    hypothesis = _parse_hypothesis(result["response"], fname, snapshot, round_num=1)

    _log(f"Hypothesis formed after {result['turns']} turns, {len(result['tool_calls'])} tool calls")
    _log(f"Strategy: {hypothesis.strategy}")
    _log(f"Expected: {hypothesis.current_complexity} -> {hypothesis.proposed_complexity}")

    # Write to conversation context if available
    if ctx:
        ctx.current_hypothesis = hypothesis.strategy
        ctx.add_message(
            sender="analyst",
            recipient="optimizer",
            message_type="hypothesis",
            content={
                "hypothesis": hypothesis.strategy,
                "current_complexity": hypothesis.current_complexity,
                "proposed_complexity": hypothesis.proposed_complexity,
                "bottleneck": hypothesis.bottleneck,
                "expected_speedup": hypothesis.expected_speedup,
                "risks": hypothesis.risks,
            },
            confidence=0.8,
        )

    if emitter:
        emitter.complete(
            EventType.ANALYSIS_COMPLETE, "analyst",
            f"Hypothesis: {hypothesis.strategy}",
            function_name=fname,
            data={
                "strategy": hypothesis.strategy,
                "current": hypothesis.current_complexity,
                "proposed": hypothesis.proposed_complexity,
                "tool_calls": len(result["tool_calls"]),
                "turns": result["turns"],
                "thinking_traces": len(result["thinking"]),
                "model": result["model"],
            }
        )

    return hypothesis


def diagnose_failure(
    snapshot: BehavioralSnapshot,
    failed_validation: ValidationResult,
    previous_hypothesis: AnalysisHypothesis,
    emitter: Optional[EventEmitter] = None,
    enable_thinking: bool = True,
    ctx: Optional[ConversationContext] = None,
) -> AnalysisHypothesis:
    """
    Round 2+: AGENTIC failure diagnosis with tool use.

    Claude investigates WHY the optimization was rejected by calling tools,
    then forms a NEW hypothesis that avoids the same failure.

    Args:
        snapshot: Behavioral snapshot of the function
        failed_validation: The validation result that rejected the optimization
        previous_hypothesis: The hypothesis that failed
        emitter: Event emitter for SSE streaming
        enable_thinking: Enable extended thinking
        ctx: Optional ConversationContext for multi-agent communication
    """
    fname = snapshot.function_name
    round_num = failed_validation.round_number + 1
    _log = lambda msg: emitter.log("analyst", msg, function_name=fname, round_number=round_num) if emitter else None

    _log(f"Round {round_num}: Diagnosing failure for {fname}()...")

    if emitter:
        emitter.complete(
            EventType.REJECTION_DIAGNOSIS, "analyst",
            f"Investigating Round {round_num-1} rejection",
            function_name=fname, round_number=round_num,
        )

    # Build failure context
    veto_str = "\n".join(f"- {v}" for v in failed_validation.veto_reasons) or "No hard vetoes"
    metrics_str = ""
    if failed_validation.improvements:
        metrics_str = "\n".join(
            f"- {'IMPROVED' if m.improved else 'REGRESSED'} {m.metric_name}: {m.before:.2f} -> {m.after:.2f} ({m.delta_percent:+.1f}%)"
            for m in failed_validation.improvements
        )

    # Build round-aware system prompt with conversation context
    system_prompt = build_analyst_system_prompt(ctx) if ctx else DIAGNOSIS_SYSTEM

    user_msg = (
        f"## REJECTED Optimization — Diagnose and propose a new strategy\n\n"
        f"### Original Function: {fname}()\n"
        f"```python\n{snapshot.source_code}\n```\n\n"
        f"### Previous Strategy (FAILED)\n"
        f"- Strategy: {previous_hypothesis.strategy}\n"
        f"- Expected: {previous_hypothesis.current_complexity} -> {previous_hypothesis.proposed_complexity}\n\n"
        f"### Rejection Reasons\n{veto_str}\n\n"
        f"### Metric Changes\n{metrics_str}\n\n"
        f"Use your tools to investigate what went wrong. Then propose a DIFFERENT strategy."
    )

    executor = _make_event_tool_executor(emitter, fname, round_num)

    def on_thinking(text):
        if emitter:
            preview = text[:200] + "..." if len(text) > 200 else text
            emitter.log("analyst", f"[thinking] {preview}",
                       function_name=fname, round_number=round_num,
                       data={"type": "thinking"})

    result = call_claude_with_tools(
        system_prompt=system_prompt,
        user_message=user_msg,
        tools=ANALYST_TOOLS,
        tool_executor=executor,
        enable_thinking=enable_thinking,
        on_thinking=on_thinking,
        max_turns=8,
    )

    hypothesis = _parse_hypothesis(result["response"], fname, snapshot, round_num)

    _log(f"New hypothesis after {len(result['tool_calls'])} tool calls")
    _log(f"Diagnosis: {hypothesis.failure_diagnosis}")
    _log(f"New strategy: {hypothesis.strategy}")

    # Write to conversation context if available
    if ctx:
        ctx.current_hypothesis = hypothesis.strategy
        ctx.add_message(
            sender="analyst",
            recipient="optimizer",
            message_type="diagnosis",
            content={
                "hypothesis": hypothesis.strategy,
                "failure_diagnosis": hypothesis.failure_diagnosis,
                "current_complexity": hypothesis.current_complexity,
                "proposed_complexity": hypothesis.proposed_complexity,
                "bottleneck": hypothesis.bottleneck,
                "previous_strategy_failed": previous_hypothesis.strategy,
            },
            confidence=0.6,
        )

    if emitter:
        emitter.complete(
            EventType.ANALYSIS_COMPLETE, "analyst",
            f"Round {round_num}: {hypothesis.strategy}",
            function_name=fname, round_number=round_num,
            data={
                "strategy": hypothesis.strategy,
                "diagnosis": hypothesis.failure_diagnosis,
                "tool_calls": len(result["tool_calls"]),
                "round": round_num,
            }
        )

    return hypothesis


# ═══════════════════════════════════════════════════════════
# Parse hypothesis from Claude's response
# ═══════════════════════════════════════════════════════════

def _parse_hypothesis(
    response_text: str,
    function_name: str,
    snapshot: BehavioralSnapshot,
    round_num: int,
) -> AnalysisHypothesis:
    """Extract AnalysisHypothesis from Claude's text response (which should contain JSON)."""

    # Try to extract JSON from the response
    text = response_text.strip()

    # Strip markdown fences
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start) if "```" in text[start:] else len(text)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start) if "```" in text[start:] else len(text)
        text = text[start:end].strip()

    # Try to find JSON object
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Find first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                data = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

    return AnalysisHypothesis(
        function_name=function_name,
        current_complexity=data.get("current_complexity", snapshot.big_o_estimate or "unknown"),
        proposed_complexity=data.get("proposed_complexity", "unknown"),
        bottleneck=data.get("bottleneck", "Could not determine"),
        strategy=data.get("strategy", "General optimization"),
        expected_speedup=data.get("expected_speedup", "unknown"),
        risks=data.get("risks", []),
        round_number=round_num,
        failure_diagnosis=data.get("failure_diagnosis"),
    )
