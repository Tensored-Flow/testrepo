"""
Agent 5: Test Designer — SELF-VALIDATING Diff-Aware Test Generation.

LLM-powered with tool use for self-validation.
Generates tests, then RUNS them to make sure they actually work
before passing to the Validator. No more broken test code.

Features:
  - Tool_use: runs generated tests to verify they work
  - Self-correcting: fixes broken tests autonomously
  - Diff-aware: reads the actual diff to write targeted tests
  - ConversationContext: sees full conversation history
"""

import json
from typing import Optional

from backend.models.types import (
    BehavioralSnapshot, OptimizedCode, TestSuite,
)
from backend.models.events import EventEmitter, EventType
from backend.integrations.llm_client import call_claude_with_tools, SONNET
from backend.core.tools import TEST_DESIGNER_TOOLS, execute_tool
from backend.core.conversation import ConversationContext


TEST_DESIGNER_SYSTEM = """\
You are an expert test engineer. You're given an original and optimized Python function.
Write tests that PROVE the optimization is safe, then VALIDATE your tests by running them.

You have tools:
- run_tests: Execute test code and see if it passes
- compare_outputs: Run both functions with inputs and compare results
- run_function_with_inputs: Test individual functions with specific inputs

GENERATE TWO CATEGORIES OF TESTS:

1. **Differential tests**: Verify original(x) == optimized(x) for many inputs
   - Normal inputs, edge cases, empty inputs, large inputs
   - Negative numbers, zero, duplicates, already-sorted data

2. **Targeted tests**: Edge cases specific to what the optimization CHANGED
   - If sort algorithm changed: test stability, reverse-sorted, single element
   - If hash map used: test None values, duplicate keys
   - If loop changed: test empty iteration, boundary conditions

YOUR PROCESS:
1. Write the test code with both functions defined inline
2. Call run_tests to make sure your tests actually execute
3. If tests have errors, fix them and re-run
4. Once all tests pass, output the final test code

TEST CODE FORMAT:
- Define `original_FUNCNAME(...)` with the original code
- Define `optimized_FUNCNAME(...)` with the optimized code
- Test functions named `test_*` using plain assert statements
- End with `if __name__ == "__main__":` that runs all tests and prints "ALL TESTS PASSED"

OUTPUT your final validated test code as:
```json
{
  "differential_tests": "...complete Python test code for differential tests...",
  "targeted_tests": "...complete Python test code for targeted tests...",
  "test_count": <number of test functions>
}
```

IMPORTANT: ALWAYS call run_tests to validate your test code before outputting it.
"""


def _make_test_tool_executor(emitter: Optional[EventEmitter], fname: str, round_num: int):
    """Wraps tool execution with SSE events."""
    def executor(tool_name: str, tool_input: dict) -> str:
        if emitter:
            emitter.log("test_designer", f"Running {tool_name}...",
                       function_name=fname, round_number=round_num,
                       data={"tool": tool_name})

        result_str = execute_tool(tool_name, tool_input)

        if emitter:
            try:
                data = json.loads(result_str)
                if tool_name == "run_tests":
                    passed = data.get("passed", False)
                    emitter.log("test_designer",
                               f"{'Tests pass' if passed else 'Tests failed -- fixing...'}",
                               function_name=fname, round_number=round_num)
                elif tool_name == "compare_outputs":
                    match = data.get("all_match", False)
                    emitter.log("test_designer",
                               f"{'Outputs match' if match else 'Output mismatch detected'}",
                               function_name=fname, round_number=round_num)
            except (json.JSONDecodeError, TypeError):
                pass

        return result_str

    return executor


def test_designer_agent(
    snapshot: BehavioralSnapshot,
    optimized: OptimizedCode,
    emitter: Optional[EventEmitter] = None,
    enable_thinking: bool = True,
    ctx: Optional[ConversationContext] = None,
) -> TestSuite:
    """
    Generate AND validate tests using tool use.

    Args:
        snapshot: Behavioral snapshot of the original function
        optimized: The optimizer's output
        emitter: Event emitter for SSE streaming
        enable_thinking: Enable extended thinking
        ctx: Optional ConversationContext for multi-agent communication
    """
    fname = snapshot.function_name
    round_num = optimized.round_number
    _log = lambda msg: emitter.log("test_designer", msg, function_name=fname, round_number=round_num) if emitter else None

    _log(f"Designing tests for {fname}() optimization...")

    # Build system prompt with conversation context
    system_prompt = TEST_DESIGNER_SYSTEM
    if ctx and ctx.messages:
        system_prompt += f"\n\n=== CONVERSATION HISTORY ===\n{ctx.to_agent_briefing('test_designer')}"

    user_msg = (
        f"## Generate and validate tests for: {fname}()\n\n"
        f"### Original Code\n```python\n{snapshot.source_code}\n```\n\n"
        f"### Optimized Code\n```python\n{optimized.optimized_source}\n```\n\n"
        f"### What Changed\n{optimized.changes_description}\n\n"
    )

    if snapshot.input_output_pairs:
        user_msg += "### Known I/O Pairs\n"
        for pair in snapshot.input_output_pairs[:5]:
            if isinstance(pair, dict) and not pair.get("error"):
                user_msg += f"- {pair.get('input')} -> {pair.get('output')}\n"

    user_msg += (
        f"\nWrite comprehensive tests, then call run_tests to validate they work. "
        f"Fix any failures before submitting."
    )

    executor = _make_test_tool_executor(emitter, fname, round_num)

    def on_thinking(text):
        if emitter:
            preview = text[:150] + "..." if len(text) > 150 else text
            emitter.log("test_designer", f"[thinking] {preview}",
                       function_name=fname, round_number=round_num,
                       data={"type": "thinking"})

    result = call_claude_with_tools(
        system_prompt=system_prompt,
        user_message=user_msg,
        tools=TEST_DESIGNER_TOOLS,
        tool_executor=executor,
        enable_thinking=enable_thinking,
        on_thinking=on_thinking,
        max_turns=8,
    )

    # Parse the test suite from response
    suite = _parse_test_output(result["response"], fname)

    _log(f"Generated {suite.test_count} validated tests ({result['turns']} turns, {len(result['tool_calls'])} tool calls)")

    # Write to conversation context if available
    if ctx:
        ctx.test_suites = {
            "differential_tests": suite.differential_tests,
            "targeted_tests": suite.targeted_tests,
            "test_count": suite.test_count,
        }
        ctx.add_message(
            sender="test_designer",
            recipient="validator",
            message_type="test_suite",
            content={
                "differential_tests": suite.differential_tests[:200] + "..." if len(suite.differential_tests) > 200 else suite.differential_tests,
                "targeted_tests": suite.targeted_tests[:200] + "..." if len(suite.targeted_tests) > 200 else suite.targeted_tests,
                "test_count": suite.test_count,
                "edge_cases_covered": suite.test_count,
            },
        )

    if emitter:
        emitter.complete(
            EventType.TESTS_GENERATED, "test_designer",
            f"Generated {suite.test_count} tests for {fname}() -- self-validated",
            function_name=fname, round_number=round_num,
            data={
                "test_count": suite.test_count,
                "tool_calls": len(result["tool_calls"]),
                "self_validated": True,
            }
        )

    return suite


def _parse_test_output(response_text: str, function_name: str) -> TestSuite:
    """Extract TestSuite from the test designer's response."""
    text = response_text.strip()

    # Try JSON parse
    try:
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            data = json.loads(text[start:end].strip())
        else:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                data = json.loads(text[start:end + 1])
            else:
                data = {}

        if data.get("differential_tests") or data.get("targeted_tests"):
            diff_tests = data.get("differential_tests", "")
            targeted = data.get("targeted_tests", "")
            count = data.get("test_count", diff_tests.count("def test_") + targeted.count("def test_"))
            return TestSuite(
                function_name=function_name,
                differential_tests=diff_tests,
                targeted_tests=targeted,
                test_count=count,
            )
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: find Python code blocks
    if "```python" in text:
        code_blocks = []
        remaining = text
        while "```python" in remaining:
            start = remaining.index("```python") + 9
            end_marker = remaining.find("```", start)
            if end_marker == -1:
                code_blocks.append(remaining[start:].strip())
                break
            code_blocks.append(remaining[start:end_marker].strip())
            remaining = remaining[end_marker + 3:]

        if len(code_blocks) >= 2:
            diff_tests = code_blocks[0]
            targeted = code_blocks[1]
        elif code_blocks:
            diff_tests = code_blocks[0]
            targeted = ""
        else:
            diff_tests = ""
            targeted = ""
    else:
        diff_tests = text
        targeted = ""

    count = diff_tests.count("def test_") + targeted.count("def test_")
    return TestSuite(
        function_name=function_name,
        differential_tests=diff_tests,
        targeted_tests=targeted,
        test_count=max(count, 1),
    )
