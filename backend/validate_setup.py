#!/usr/bin/env python3
"""
Validate Setup — Pre-flight checks for the ComplexityImprover system.

Run this before starting the server to catch configuration issues early:
    python -m backend.validate_setup

Checks:
  1. All module imports resolve
  2. Sandbox backend is healthy (subprocess or Docker)
  3. Tool handler coverage (every tool schema has a handler)
  4. API key is set (ANTHROPIC_API_KEY)
  5. Required packages installed (lizard, radon, numpy, scipy)
  6. ConversationContext round-trips correctly
  7. EventEmitter works
"""

import sys
import os
import importlib

# ─────────────────────────────────────────────
# Test harness
# ─────────────────────────────────────────────

_pass_count = 0
_fail_count = 0
_warnings = []


def check(name: str, condition: bool, detail: str = ""):
    global _pass_count, _fail_count
    if condition:
        _pass_count += 1
        print(f"  [PASS] {name}")
    else:
        _fail_count += 1
        msg = f"  [FAIL] {name}"
        if detail:
            msg += f" -- {detail}"
        print(msg)


def warn(name: str, detail: str):
    _warnings.append((name, detail))
    print(f"  [WARN] {name} -- {detail}")


# ─────────────────────────────────────────────
# 1. Module imports
# ─────────────────────────────────────────────

print("\n=== 1. Module Imports ===")

REQUIRED_MODULES = [
    "backend",
    "backend.models",
    "backend.models.types",
    "backend.models.events",
    "backend.core",
    "backend.core.sandbox_interface",
    "backend.core.sandbox",
    "backend.core.sandbox_factory",
    "backend.core.docker_sandbox",
    "backend.core.conversation",
    "backend.core.tools",
    "backend.integrations",
    "backend.integrations.llm_client",
    "backend.agents",
    "backend.agents.triage",
    "backend.agents.snapshot",
    "backend.agents.analyst",
    "backend.agents.optimizer",
    "backend.agents.test_designer",
    "backend.agents.validator",
    "backend.agents.planner",
    "backend.agents.report_generator",
]

for mod in REQUIRED_MODULES:
    try:
        importlib.import_module(mod)
        check(f"import {mod}", True)
    except Exception as e:
        check(f"import {mod}", False, str(e))


# ─────────────────────────────────────────────
# 2. Key type imports
# ─────────────────────────────────────────────

print("\n=== 2. Key Type Imports ===")

try:
    from backend.models.types import (
        Category, Verdict, TargetFunction, TriageResult,
        BehavioralSnapshot, BenchmarkPoint, StaticMetrics,
        AnalysisHypothesis, OptimizedCode, TestSuite,
        TestResults, MetricsComparison, ValidationResult,
        PipelineResult, RunResult,
    )
    check("All types import", True)
except ImportError as e:
    check("All types import", False, str(e))

try:
    from backend.models.events import EventType, EventEmitter, PipelineEvent
    check("Event types import", True)
    check("AGENT_HANDOFF event exists", hasattr(EventType, "AGENT_HANDOFF"))
except ImportError as e:
    check("Event types import", False, str(e))

try:
    from backend.core.conversation import ConversationContext, AgentMessage
    check("ConversationContext import", True)
except ImportError as e:
    check("ConversationContext import", False, str(e))

try:
    from backend.core.sandbox_interface import SandboxBackend, SandboxResult
    check("Sandbox interface import", True)
except ImportError as e:
    check("Sandbox interface import", False, str(e))


# ─────────────────────────────────────────────
# 3. Tool handler coverage
# ─────────────────────────────────────────────

print("\n=== 3. Tool Handler Coverage ===")

try:
    from backend.core.tools import (
        ANALYST_TOOLS, OPTIMIZER_TOOLS, TEST_DESIGNER_TOOLS,
        TOOL_HANDLERS, execute_tool,
    )

    # Collect all tool names referenced in schemas
    all_tool_names = set()
    for tools_list in [ANALYST_TOOLS, OPTIMIZER_TOOLS, TEST_DESIGNER_TOOLS]:
        for tool in tools_list:
            name = tool.get("name", "")
            if name:
                all_tool_names.add(name)

    # Check each has a handler
    for name in sorted(all_tool_names):
        has_handler = name in TOOL_HANDLERS
        check(f"Handler for '{name}'", has_handler,
              "No handler registered" if not has_handler else "")

    # Check no orphaned handlers
    orphaned = set(TOOL_HANDLERS.keys()) - all_tool_names
    if orphaned:
        warn("Orphaned handlers", f"Handlers without tool schemas: {orphaned}")
    else:
        check("No orphaned handlers", True)

except Exception as e:
    check("Tool handler coverage", False, str(e))


# ─────────────────────────────────────────────
# 4. API key
# ─────────────────────────────────────────────

print("\n=== 4. API Key ===")

api_key = os.environ.get("ANTHROPIC_API_KEY", "")
if api_key:
    check("ANTHROPIC_API_KEY set", True)
    check("API key format", api_key.startswith("sk-ant-"), f"Unexpected prefix: {api_key[:8]}...")
else:
    warn("ANTHROPIC_API_KEY", "Not set -- LLM agents will fail")


# ─────────────────────────────────────────────
# 5. Required packages
# ─────────────────────────────────────────────

print("\n=== 5. Required Packages ===")

REQUIRED_PACKAGES = {
    "anthropic": "anthropic",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "numpy": "numpy",
    "scipy": "scipy",
}

OPTIONAL_PACKAGES = {
    "lizard": "lizard",
    "radon": "radon",
}

for display, pkg in REQUIRED_PACKAGES.items():
    try:
        importlib.import_module(pkg)
        check(f"Package: {display}", True)
    except ImportError:
        check(f"Package: {display}", False, f"pip install {pkg}")

for display, pkg in OPTIONAL_PACKAGES.items():
    try:
        importlib.import_module(pkg)
        check(f"Package: {display}", True)
    except ImportError:
        warn(f"Package: {display}", f"pip install {pkg} (needed for metrics)")


# ─────────────────────────────────────────────
# 6. ConversationContext round-trip
# ─────────────────────────────────────────────

print("\n=== 6. ConversationContext ===")

try:
    from backend.core.conversation import ConversationContext

    ctx = ConversationContext(
        function_name="test_func",
        original_source="def test_func(x): return x",
        file_path="test.py",
        category="A",
    )

    # Add a message
    msg = ctx.add_message(
        sender="analyst",
        recipient="optimizer",
        message_type="hypothesis",
        content={"strategy": "use hash map"},
        confidence=0.85,
    )
    check("add_message()", msg is not None and msg.sender == "analyst")

    # Query messages
    for_optimizer = ctx.get_messages_for("optimizer")
    check("get_messages_for()", len(for_optimizer) == 1)

    from_analyst = ctx.get_messages_from("analyst")
    check("get_messages_from()", len(from_analyst) == 1)

    # Briefing
    briefing = ctx.to_agent_briefing("optimizer")
    check("to_agent_briefing()", "analyst" in briefing and "hash map" in briefing)

    # Rejection summary
    ctx.rejection_history.append({
        "round": 1,
        "vetoes_triggered": ["runtime regression"],
        "summary": "too slow",
    })
    summary = ctx.get_rejection_summary()
    check("get_rejection_summary()", "runtime regression" in summary)

except Exception as e:
    check("ConversationContext round-trip", False, str(e))


# ─────────────────────────────────────────────
# 7. EventEmitter
# ─────────────────────────────────────────────

print("\n=== 7. EventEmitter ===")

try:
    from backend.models.events import EventEmitter, EventType

    emitter = EventEmitter()
    received = []
    emitter.on_event(lambda e: received.append(e))

    emitter.log("test", "hello")
    check("EventEmitter.log()", len(received) == 1)

    emitter.complete(EventType.AGENT_HANDOFF, "analyst", "handoff test",
                     data={"from": "analyst", "to": "optimizer"})
    check("EventEmitter AGENT_HANDOFF", len(received) == 2)
    check("AGENT_HANDOFF payload", received[1].data.get("from") == "analyst")

    # SSE serialization
    sse = received[0].to_sse()
    check("PipelineEvent.to_sse()", sse.startswith("data: ") and sse.endswith("\n\n"))

except Exception as e:
    check("EventEmitter", False, str(e))


# ─────────────────────────────────────────────
# 8. Sandbox health
# ─────────────────────────────────────────────

print("\n=== 8. Sandbox ===")

try:
    from backend.core.sandbox_factory import create_sandbox
    sandbox = create_sandbox()
    check("create_sandbox()", sandbox is not None)
    check("Sandbox type", True, detail=type(sandbox).__name__)
except Exception as e:
    check("Sandbox factory", False, str(e))


# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────

print(f"\n{'='*50}")
print(f"  PASS: {_pass_count}  |  FAIL: {_fail_count}  |  WARN: {len(_warnings)}")
print(f"{'='*50}")

if _fail_count > 0:
    print("\nFailed checks must be fixed before running the pipeline.")
    sys.exit(1)
elif _warnings:
    print("\nWarnings are non-blocking but may cause runtime issues.")
    sys.exit(0)
else:
    print("\nAll checks passed. System ready.")
    sys.exit(0)
