"""
Agent 4: Optimizer — SELF-VALIDATING Code Generation.

Claude generates optimized code AND validates it using tools.
Instead of blind generation -> external validation, the Optimizer:
  1. Generates optimized code
  2. Calls compile_check to verify syntax
  3. Calls compare_outputs to verify behavioral equivalence
  4. If self-validation fails, fixes the code autonomously
  5. Optionally calls quick_benchmark to verify speedup

This self-correcting loop is visible in the agent feed — judges can
see Claude catch its own mistakes and fix them.

Features:
  - Tool_use for self-validation (compile_check, compare_outputs, run_tests)
  - Extended thinking for code generation reasoning
  - Self-correcting: fixes own bugs before submitting
  - Multi-turn: may take several turns to get code right
  - ConversationContext: sees full conversation history
  - Web research: can look up optimization approaches if stuck
"""

import json
from typing import Optional

from backend.models.types import (
    BehavioralSnapshot, AnalysisHypothesis, OptimizedCode,
)
from backend.models.events import EventEmitter, EventType
from backend.integrations.llm_client import call_claude_with_tools, SONNET
from backend.core.tools import OPTIMIZER_TOOLS, execute_tool
from backend.core.conversation import ConversationContext


# ═══════════════════════════════════════════════════════════
# System Prompt
# ═══════════════════════════════════════════════════════════

OPTIMIZER_SYSTEM = """\
You are an expert Python optimizer. You receive a function and an optimization strategy,
and you must produce an optimized version that is PROVEN correct.

You have tools to validate your work:
- compile_check: Verify your code parses and the function can be extracted
- compare_outputs: Run both original and optimized with the same inputs to verify they produce identical results
- run_tests: Execute test code to verify correctness
- quick_benchmark: Compare performance of original vs optimized
- web_research: Search for optimization techniques if you need ideas

YOUR PROCESS:
1. Generate the optimized function based on the strategy
2. Call compile_check to verify it's valid Python
3. Call compare_outputs with diverse test inputs to verify behavioral equivalence
4. If anything fails, FIX your code and re-validate
5. Once validated, output the final optimized code

OUTPUT FORMAT:
After you've validated your code with tools, output your final answer containing:
```json
{
  "optimized_source": "def function_name(...):\\n    ...",
  "changes_description": "What you changed and why",
  "self_validation": {
    "syntax_valid": true/false,
    "outputs_match": true/false,
    "tests_passed": true/false
  }
}
```

RULES:
1. Keep the EXACT same function signature
2. Produce IDENTICAL outputs for ALL inputs
3. Do NOT use: os, sys, subprocess, eval, exec, __import__, open, pickle
4. You may use: math, collections, itertools, functools, heapq, bisect, operator
5. Put imports INSIDE the function body if needed
6. ALWAYS validate with compare_outputs before submitting — never submit untested code
"""


# ═══════════════════════════════════════════════════════════
# Event-emitting tool executor
# ═══════════════════════════════════════════════════════════

def _make_optimizer_tool_executor(emitter: Optional[EventEmitter], fname: str, round_num: int):
    """Wraps tool execution with SSE events for the agent feed."""
    def executor(tool_name: str, tool_input: dict) -> str:
        if emitter:
            if tool_name == "compile_check":
                emitter.log("optimizer", f"Checking syntax...",
                           function_name=fname, round_number=round_num,
                           data={"tool": tool_name})
            elif tool_name == "compare_outputs":
                n = len(tool_input.get("test_inputs", []))
                emitter.log("optimizer", f"Testing behavioral equivalence ({n} inputs)...",
                           function_name=fname, round_number=round_num,
                           data={"tool": tool_name})
            elif tool_name == "run_tests":
                emitter.log("optimizer", f"Running validation tests...",
                           function_name=fname, round_number=round_num,
                           data={"tool": tool_name})
            elif tool_name == "quick_benchmark":
                emitter.log("optimizer", f"Benchmarking original vs optimized...",
                           function_name=fname, round_number=round_num,
                           data={"tool": tool_name})
            elif tool_name == "web_research":
                emitter.log("optimizer", f"Researching: {tool_input.get('query', '')[:50]}...",
                           function_name=fname, round_number=round_num,
                           data={"tool": tool_name})

        result_str = execute_tool(tool_name, tool_input)

        if emitter:
            try:
                data = json.loads(result_str)
                if tool_name == "compile_check":
                    valid = data.get("valid", False)
                    emitter.log("optimizer",
                               f"{'Syntax valid' if valid else 'Syntax error: ' + data.get('error', '')}",
                               function_name=fname, round_number=round_num)
                elif tool_name == "compare_outputs":
                    if data.get("all_match"):
                        emitter.log("optimizer",
                                   f"All {data.get('total_tests', '?')} outputs match original",
                                   function_name=fname, round_number=round_num)
                    else:
                        emitter.log("optimizer",
                                   f"{data.get('mismatches', '?')} output mismatches -- fixing...",
                                   function_name=fname, round_number=round_num)
                elif tool_name == "quick_benchmark":
                    speedup = data.get("speedup", 0)
                    emitter.log("optimizer",
                               f"Speedup: {speedup:.1f}x {'(faster!)' if speedup > 1 else '(slower)'}",
                               function_name=fname, round_number=round_num)
            except (json.JSONDecodeError, TypeError):
                pass

        return result_str

    return executor


# ═══════════════════════════════════════════════════════════
# Build the user message
# ═══════════════════════════════════════════════════════════

def _build_optimizer_message(
    snapshot: BehavioralSnapshot,
    hypothesis: AnalysisHypothesis,
    ctx: Optional[ConversationContext] = None,
) -> str:
    """Build the user prompt with function + strategy + test inputs for validation."""

    # Generate some test inputs the optimizer can use for compare_outputs
    test_inputs = _generate_validation_inputs(snapshot)
    test_inputs_json = json.dumps(test_inputs, default=str)

    parts = [
        f"## Optimize: {snapshot.function_name}()",
        f"\n### Original Code\n```python\n{snapshot.source_code}\n```",
        f"\n### Optimization Strategy (from Analyst)",
        f"- **Bottleneck:** {hypothesis.bottleneck}",
        f"- **Strategy:** {hypothesis.strategy}",
        f"- **Current complexity:** {hypothesis.current_complexity}",
        f"- **Target complexity:** {hypothesis.proposed_complexity}",
    ]

    if hypothesis.risks:
        parts.append(f"- **Risks:** {', '.join(hypothesis.risks)}")

    if hypothesis.round_number > 1 and hypothesis.failure_diagnosis:
        parts.append(f"\n### Previous Attempt Failed (Round {hypothesis.round_number - 1})")
        parts.append(f"**Diagnosis:** {hypothesis.failure_diagnosis}")
        parts.append("Your code MUST avoid this failure mode.")

    # Add conversation context if available
    if ctx and ctx.messages:
        parts.append(f"\n### Conversation History")
        parts.append(ctx.to_agent_briefing("optimizer"))

    parts.append(f"\n### Test Inputs for Validation")
    parts.append(f"Use these with compare_outputs to verify your optimization:")
    parts.append(f"```json\n{test_inputs_json}\n```")

    parts.append(f"\nGenerate the optimized function, then validate it with tools before submitting.")

    return "\n".join(parts)


def _generate_validation_inputs(snapshot: BehavioralSnapshot) -> list:
    """Generate test inputs the optimizer can use for self-validation."""
    # Use inputs from the snapshot if available
    if snapshot.input_output_pairs:
        inputs = []
        for pair in snapshot.input_output_pairs:
            if isinstance(pair, dict) and pair.get("input") and not pair.get("error"):
                inp = pair["input"]
                if isinstance(inp, dict) and "args" in inp:
                    inputs.append(inp)
        if inputs:
            return inputs[:8]

    # Fallback: generate generic inputs
    fname = snapshot.function_name.lower()
    list_keywords = ["sort", "search", "find", "filter", "merge", "process",
                     "reverse", "rotate", "partition", "flatten", "max", "min", "sum"]

    if any(kw in fname for kw in list_keywords):
        return [
            {"args": [[3, 1, 4, 1, 5, 9, 2, 6]], "kwargs": {}},
            {"args": [[1]], "kwargs": {}},
            {"args": [[]], "kwargs": {}},
            {"args": [[5, 4, 3, 2, 1]], "kwargs": {}},
            {"args": [[1, 1, 1, 1]], "kwargs": {}},
            {"args": [[-3, -1, -4, -1, -5]], "kwargs": {}},
            {"args": [[42]], "kwargs": {}},
            {"args": [list(range(20, 0, -1))], "kwargs": {}},
        ]
    else:
        return [
            {"args": [0], "kwargs": {}},
            {"args": [1], "kwargs": {}},
            {"args": [-1], "kwargs": {}},
            {"args": [42], "kwargs": {}},
            {"args": [100], "kwargs": {}},
        ]


# ═══════════════════════════════════════════════════════════
# Main Agent Entry Point
# ═══════════════════════════════════════════════════════════

def optimizer_agent(
    snapshot: BehavioralSnapshot,
    hypothesis: AnalysisHypothesis,
    emitter: Optional[EventEmitter] = None,
    enable_thinking: bool = True,
    ctx: Optional[ConversationContext] = None,
    config=None,
) -> Optional[OptimizedCode]:
    """
    SELF-VALIDATING code generation.

    Claude generates code, validates with tools, self-corrects if needed.
    All visible in the agent feed.

    Args:
        snapshot: Behavioral snapshot of the function
        hypothesis: The analyst's optimization hypothesis
        emitter: Event emitter for SSE streaming
        enable_thinking: Enable extended thinking
        ctx: Optional ConversationContext for multi-agent communication
        config: Optional PipelineConfig for demo_mode adjustments
    """
    fname = snapshot.function_name
    round_num = hypothesis.round_number
    demo_mode = config.demo_mode if config else False
    _log = lambda msg: emitter.log("optimizer", msg, function_name=fname, round_number=round_num) if emitter else None

    _log(f"Optimizer generating code for {fname}()..." + (" [DEMO]" if demo_mode else ""))
    _log(f"Strategy: {hypothesis.strategy}")

    # Build system prompt with conversation context
    system_prompt = OPTIMIZER_SYSTEM
    if ctx and ctx.messages:
        system_prompt += f"\n\n=== CONVERSATION HISTORY ===\n{ctx.to_agent_briefing('optimizer')}"

    # Demo mode: add efficiency instructions
    if demo_mode:
        system_prompt += (
            "\n\nIMPORTANT — DEMO MODE: Be efficient. "
            "Self-validate with at most 3 test inputs. "
            "Call compile_check once, compare_outputs once with a small set of inputs, then submit."
        )

    user_msg = _build_optimizer_message(snapshot, hypothesis, ctx)
    executor = _make_optimizer_tool_executor(emitter, fname, round_num)

    def on_thinking(text):
        if emitter:
            preview = text[:200] + "..." if len(text) > 200 else text
            emitter.log("optimizer", f"[thinking] {preview}",
                       function_name=fname, round_number=round_num,
                       data={"type": "thinking"})

    max_turns = config.optimizer_max_validations if config else 12

    # THE SELF-VALIDATING LOOP
    result = call_claude_with_tools(
        system_prompt=system_prompt,
        user_message=user_msg,
        tools=OPTIMIZER_TOOLS,
        tool_executor=executor,
        enable_thinking=enable_thinking,
        on_thinking=on_thinking,
        max_turns=max_turns,
    )

    # Parse the optimized code from Claude's response
    optimized = _parse_optimizer_output(result["response"], snapshot, hypothesis)

    if optimized is None:
        _log(f"Optimizer failed to produce valid code after {result['turns']} turns")
        if emitter:
            emitter.error("optimizer", f"Code generation failed for {fname}()",
                         function_name=fname, round_number=round_num)
        return None

    _log(f"Optimized code generated and self-validated ({result['turns']} turns, {len(result['tool_calls'])} tool calls)")

    # Write to conversation context if available
    if ctx:
        ctx.optimized_code = optimized.optimized_source
        ctx.add_message(
            sender="optimizer",
            recipient="validator",
            message_type="code_submission",
            content={
                "code": optimized.optimized_source,
                "changes_description": optimized.changes_description,
                "self_validation_results": {
                    "turns": result["turns"],
                    "tool_calls": len(result["tool_calls"]),
                    "tools_used": list(set(tc["tool_name"] for tc in result["tool_calls"])),
                },
            },
        )

    if emitter:
        emitter.complete(
            EventType.OPTIMIZATION_COMPLETE, "optimizer",
            f"Code generated for {fname}() -- self-validated",
            function_name=fname, round_number=round_num,
            data={
                "turns": result["turns"],
                "tool_calls": len(result["tool_calls"]),
                "self_validated": True,
                "tools_used": list(set(tc["tool_name"] for tc in result["tool_calls"])),
            }
        )

    return optimized


# ═══════════════════════════════════════════════════════════
# Parse optimizer output
# ═══════════════════════════════════════════════════════════

def _parse_optimizer_output(
    response_text: str,
    snapshot: BehavioralSnapshot,
    hypothesis: AnalysisHypothesis,
) -> Optional[OptimizedCode]:
    """Extract OptimizedCode from the optimizer's response."""
    text = response_text.strip()

    # Try JSON parse first
    try:
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            json_str = text[start:end].strip()
        else:
            start = text.find("{")
            end = text.rfind("}")
            json_str = text[start:end + 1] if start != -1 and end != -1 else ""

        if json_str:
            data = json.loads(json_str)
            source = data.get("optimized_source", "")
            if source and f"def {snapshot.function_name}" in source:
                return OptimizedCode(
                    function_name=snapshot.function_name,
                    original_source=snapshot.source_code,
                    optimized_source=source,
                    changes_description=data.get("changes_description", hypothesis.strategy),
                    strategy_used=hypothesis.strategy,
                    round_number=hypothesis.round_number,
                )
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: find Python code block
    if "```python" in text:
        try:
            start = text.index("```python") + 9
            end = text.index("```", start)
            code = text[start:end].strip()
            if f"def {snapshot.function_name}" in code:
                return OptimizedCode(
                    function_name=snapshot.function_name,
                    original_source=snapshot.source_code,
                    optimized_source=code,
                    changes_description=hypothesis.strategy,
                    strategy_used=hypothesis.strategy,
                    round_number=hypothesis.round_number,
                )
        except ValueError:
            pass

    # Last resort: find def statement directly
    fname = snapshot.function_name
    if f"def {fname}" in text:
        start = text.index(f"def {fname}")
        lines = text[start:].split("\n")
        func_lines = [lines[0]]
        for line in lines[1:]:
            if line.strip() == "" or line[0] == " " or line[0] == "\t":
                func_lines.append(line)
            elif line.startswith("def ") or line.startswith("class ") or line.startswith("#"):
                break
            else:
                func_lines.append(line)

        code = "\n".join(func_lines).strip()
        if code:
            return OptimizedCode(
                function_name=fname,
                original_source=snapshot.source_code,
                optimized_source=code,
                changes_description=hypothesis.strategy,
                strategy_used=hypothesis.strategy,
                round_number=hypothesis.round_number,
            )

    return None
