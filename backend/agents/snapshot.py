"""
Agent 2: Snapshot — Behavioral fingerprinting.

Deterministic agent (no LLM). Captures the "before" state of each
Category A function: static metrics, benchmarks, Big O estimate,
input→output pairs, and bytecode analysis.

The Validator (Agent 6) later captures "after" and compares.
"""

import math
import textwrap
from typing import Optional

try:
    import lizard
except ImportError:
    lizard = None

try:
    import radon.metrics
    import radon.complexity
    import radon.raw
except ImportError:
    radon = None

try:
    import numpy as np
except ImportError:
    np = None

from backend.models.types import (
    TargetFunction, BehavioralSnapshot, StaticMetrics, BenchmarkPoint,
)
from backend.models.events import EventEmitter, EventType
from backend.core.sandbox import (
    run_function_with_inputs, benchmark_function, compile_function,
)


# ─────────────────────────────────────────────
# Static Metrics (lizard + radon + raw)
# ─────────────────────────────────────────────

def compute_static_metrics(source_code: str, file_path: str = "<string>") -> StaticMetrics:
    """Compute all static metrics for a single function's source code."""
    metrics = StaticMetrics()
    lines = source_code.strip().split("\n")
    metrics.loc = len(lines)

    # Count parameters from first line (rough but fast)
    first_line = lines[0] if lines else ""
    if "(" in first_line and ")" in first_line:
        params = first_line.split("(", 1)[1].rsplit(")", 1)[0]
        metrics.parameter_count = len([p for p in params.split(",") if p.strip() and p.strip() != "self"])

    # Nesting depth (count max indentation)
    max_indent = 0
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            indent = len(line) - len(stripped)
            max_indent = max(max_indent, indent)
    metrics.nesting_depth = max_indent // 4  # assume 4-space indent

    # Lizard: cyclomatic + cognitive complexity
    if lizard:
        try:
            analysis = lizard.analyze_file.analyze_source_code(file_path, source_code)
            if analysis.function_list:
                func = analysis.function_list[0]
                metrics.cyclomatic_complexity = func.cyclomatic_complexity
        except Exception:
            pass

    # Radon: Halstead + Maintainability Index
    if radon:
        try:
            halstead = radon.metrics.h_visit(source_code)
            if halstead:
                h = halstead[0] if isinstance(halstead, list) else halstead
                metrics.halstead_volume = getattr(h, 'volume', 0.0) or 0.0
                metrics.halstead_difficulty = getattr(h, 'difficulty', 0.0) or 0.0
                metrics.halstead_effort = getattr(h, 'effort', 0.0) or 0.0
                metrics.halstead_bugs = getattr(h, 'bugs', 0.0) or 0.0
        except Exception:
            pass

        try:
            mi = radon.metrics.mi_visit(source_code, True)
            metrics.maintainability_index = mi if isinstance(mi, (int, float)) else 0.0
        except Exception:
            pass

    # Cognitive complexity (rough estimate: count nesting + branching)
    keywords = ["if ", "elif ", "else:", "for ", "while ", "except ", "with "]
    cc = 0
    for line in lines:
        stripped = line.strip()
        indent_level = (len(line) - len(stripped)) // 4
        for kw in keywords:
            if stripped.startswith(kw):
                cc += 1 + indent_level  # nesting increments add to cognitive load
                break
    metrics.cognitive_complexity = cc

    return metrics


# ─────────────────────────────────────────────
# Bytecode Analysis (dis module)
# ─────────────────────────────────────────────

def analyze_bytecode(source_code: str, function_name: str) -> tuple[int, dict]:
    """
    Analyze CPython bytecode of a function.
    Returns (instruction_count, category_counts).
    """
    try:
        func = compile_function(source_code, function_name)
        if func is None:
            return 0, {}

        import dis
        instructions = list(dis.get_instructions(func))
        count = len(instructions)

        categories = {}
        for instr in instructions:
            cat = _categorize_opcode(instr.opname)
            categories[cat] = categories.get(cat, 0) + 1

        return count, categories
    except Exception:
        return 0, {}


def _categorize_opcode(opname: str) -> str:
    """Categorize a CPython opcode into a human-readable bucket."""
    if "LOAD" in opname or "STORE" in opname:
        return "memory"
    elif "CALL" in opname:
        return "calls"
    elif "JUMP" in opname or "POP_JUMP" in opname:
        return "branches"
    elif "COMPARE" in opname:
        return "comparisons"
    elif "BINARY" in opname or "UNARY" in opname or "INPLACE" in opname:
        return "arithmetic"
    elif "BUILD" in opname or "UNPACK" in opname:
        return "data_structures"
    elif "RETURN" in opname or "YIELD" in opname:
        return "control_flow"
    elif "FOR" in opname or "GET_ITER" in opname:
        return "iteration"
    else:
        return "other"


# ─────────────────────────────────────────────
# Big O Estimation
# ─────────────────────────────────────────────

def estimate_big_o(benchmarks: list[BenchmarkPoint]) -> tuple[str, float]:
    """
    Estimate Big O from benchmark data using log-log regression.
    Returns (complexity_string, slope).
    """
    valid = [b for b in benchmarks if b.mean_time > 0 and b.input_size > 0]
    if len(valid) < 2:
        return "unknown", 0.0

    if np is None:
        # Fallback: simple ratio estimate
        b1, b2 = valid[0], valid[-1]
        if b1.mean_time > 0:
            ratio = b2.mean_time / b1.mean_time
            size_ratio = b2.input_size / b1.input_size
            if size_ratio > 0:
                slope = math.log(ratio) / math.log(size_ratio)
                return _slope_to_big_o(slope), slope
        return "unknown", 0.0

    log_sizes = np.log(np.array([b.input_size for b in valid], dtype=float))
    log_times = np.log(np.array([b.mean_time for b in valid], dtype=float))

    # Linear regression on log-log scale: log(T) = slope * log(N) + intercept
    coeffs = np.polyfit(log_sizes, log_times, 1)
    slope = coeffs[0]

    return _slope_to_big_o(slope), float(slope)


def _slope_to_big_o(slope: float) -> str:
    """Map log-log regression slope to Big O notation."""
    if slope < 0.1:
        return "O(1)"
    elif slope < 0.6:
        return "O(log n)"
    elif slope < 1.2:
        return "O(n)"
    elif slope < 1.6:
        return "O(n log n)"
    elif slope < 2.2:
        return "O(n^2)"
    elif slope < 3.2:
        return "O(n^3)"
    else:
        return f"O(n^{slope:.1f})"


# ─────────────────────────────────────────────
# Auto-generate test inputs for a function
# ─────────────────────────────────────────────

def generate_test_inputs(function_name: str, parameters: list[str], source_code: str) -> list[dict]:
    """
    Generate a basic set of test inputs for behavioral snapshot.
    Tries to infer parameter types from source code and names.
    Falls back to simple int/list/string inputs.
    """
    inputs = []
    param_count = len([p for p in parameters if p != "self"])

    if param_count == 0:
        # No-arg function
        inputs.append({"args": [], "kwargs": {}})
        return inputs

    # Heuristic: if function looks like it takes a list/array (sort, search, process)
    list_keywords = ["sort", "search", "find", "filter", "merge", "process",
                     "reverse", "rotate", "partition", "max", "min", "sum",
                     "average", "median", "unique", "duplicate", "flatten"]
    takes_list = any(kw in function_name.lower() for kw in list_keywords)

    if takes_list and param_count == 1:
        # Single list parameter
        test_cases = [
            [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5],
            [1],
            [],
            list(range(100, 0, -1)),
            [42] * 10,
            [-5, -3, -1, 0, 1, 3, 5],
        ]
        for case in test_cases:
            inputs.append({"args": [case], "kwargs": {}})
    elif param_count == 1:
        # Single parameter — try various types
        test_values = [0, 1, -1, 42, 100, "hello", "", [1, 2, 3]]
        for val in test_values:
            inputs.append({"args": [val], "kwargs": {}})
    elif param_count == 2:
        # Two params — common patterns
        pairs = [
            (1, 2), (0, 0), (-1, 1), (100, 200),
            ([1, 2, 3], [4, 5, 6]),
            ("hello", "world"),
        ]
        for a, b in pairs:
            inputs.append({"args": [a, b], "kwargs": {}})
    else:
        # Generic: provide ints
        inputs.append({"args": list(range(param_count)), "kwargs": {}})
        inputs.append({"args": [0] * param_count, "kwargs": {}})
        inputs.append({"args": [1] * param_count, "kwargs": {}})

    return inputs


def generate_input_generator_code(function_name: str, parameters: list[str]) -> str:
    """
    Generate a `generate_input(n)` function for benchmarking.
    Returns source code string.
    """
    list_keywords = ["sort", "search", "find", "filter", "merge", "process",
                     "reverse", "rotate", "partition", "flatten"]
    takes_list = any(kw in function_name.lower() for kw in list_keywords)

    param_count = len([p for p in parameters if p != "self"])

    if takes_list and param_count == 1:
        return "import random\ndef generate_input(n):\n    return ([random.randint(0, n*10) for _ in range(n)],)"
    elif param_count == 1:
        return "def generate_input(n):\n    return (n,)"
    elif param_count == 2:
        return "import random\ndef generate_input(n):\n    return ([random.randint(0, n) for _ in range(n)], [random.randint(0, n) for _ in range(n)])"
    else:
        args = ", ".join(["n"] * param_count)
        return f"def generate_input(n):\n    return ({args},)"


# ─────────────────────────────────────────────
# Main Agent Entry Point
# ─────────────────────────────────────────────

def snapshot_agent(
    target: TargetFunction,
    emitter: Optional[EventEmitter] = None,
    skip_benchmarks: bool = False,
) -> BehavioralSnapshot:
    """
    Capture the full behavioral snapshot of a Category A function.

    Args:
        target: The TargetFunction from triage
        emitter: Optional event emitter for SSE streaming
        skip_benchmarks: If True, skip timing benchmarks (faster for demo)

    Returns:
        BehavioralSnapshot with all metrics populated
    """
    fname = target.name
    _log = lambda msg: emitter.log("snapshot", msg, function_name=fname) if emitter else None

    _log(f"Capturing snapshot of {fname}()...")

    # 1. Static metrics
    _log(f"Computing static metrics for {fname}()...")
    static_metrics = compute_static_metrics(target.source_code, target.file_path)
    static_metrics.parameter_count = len([p for p in target.parameters if p != "self"])

    # 2. Bytecode analysis
    _log(f"Analyzing bytecode for {fname}()...")
    bc_count, bc_categories = analyze_bytecode(target.source_code, fname)

    # 3. Input/output pairs (behavioral fingerprint)
    _log(f"Generating test inputs for {fname}()...")
    test_inputs = generate_test_inputs(fname, target.parameters, target.source_code)
    io_pairs = run_function_with_inputs(
        target.source_code, fname, test_inputs, timeout=10.0
    )

    # 4. Benchmarks + Big O
    benchmarks = []
    big_o = "unknown"
    big_o_slope = 0.0

    if not skip_benchmarks:
        _log(f"Benchmarking {fname}() at multiple input sizes...")
        gen_code = generate_input_generator_code(fname, target.parameters)
        raw_benchmarks = benchmark_function(
            target.source_code, fname, gen_code,
            sizes=[100, 500, 1000, 2000],
            runs_per_size=3,
            timeout=30.0,
        )

        for b in raw_benchmarks:
            if isinstance(b, dict) and b.get("mean_time", -1) > 0:
                benchmarks.append(BenchmarkPoint(
                    input_size=b["size"],
                    mean_time=b["mean_time"],
                    std_time=b.get("std_time", 0.0),
                    memory_bytes=b.get("memory_bytes", 0),
                ))

        big_o, big_o_slope = estimate_big_o(benchmarks)
        _log(f"Estimated Big O for {fname}(): {big_o} (slope={big_o_slope:.2f})")
    else:
        _log(f"Skipping benchmarks for {fname}() (demo mode)")

    snapshot = BehavioralSnapshot(
        function_name=fname,
        file_path=target.file_path,
        source_code=target.source_code,
        static_metrics=static_metrics,
        benchmarks=benchmarks,
        big_o_estimate=big_o,
        big_o_slope=big_o_slope,
        input_output_pairs=io_pairs,
        bytecode_instruction_count=bc_count,
        bytecode_categories=bc_categories,
    )

    if emitter:
        emitter.complete(
            EventType.SNAPSHOT_COMPLETE, "snapshot",
            f"Snapshot complete for {fname}(): CC={static_metrics.cyclomatic_complexity}, "
            f"MI={static_metrics.maintainability_index:.1f}, Big O={big_o}",
            function_name=fname,
            data={
                "cyclomatic_complexity": static_metrics.cyclomatic_complexity,
                "maintainability_index": static_metrics.maintainability_index,
                "big_o": big_o,
                "bytecode_instructions": bc_count,
                "io_pairs_captured": len(io_pairs),
            }
        )

    return snapshot


# ─────────────────────────────────────────────
# Category B: Lightweight (Metrics-Only) Snapshot
# ─────────────────────────────────────────────

def lightweight_snapshot(
    target: TargetFunction,
    emitter: Optional[EventEmitter] = None,
) -> dict:
    """
    Quick metrics-only snapshot for Category B functions.
    No benchmarking, no I/O pairs (can't safely run functions with side effects).
    Just static metrics from lizard + radon + bytecode analysis.

    Args:
        target: A Category B TargetFunction (has side effects)
        emitter: Optional event emitter for SSE

    Returns:
        dict with static metrics and bytecode analysis
    """
    fname = target.name
    _log = lambda msg: emitter.log("snapshot", msg, function_name=fname) if emitter else None

    _log(f"📏 Lightweight metrics for {fname}() (Category B — no execution)")

    # 1. Static metrics (safe — no execution)
    static_metrics = compute_static_metrics(target.source_code, target.file_path)
    static_metrics.parameter_count = len([p for p in target.parameters if p != "self"])

    # 2. Bytecode analysis (safe — just disassembles, doesn't execute)
    bc_count, bc_categories = analyze_bytecode(target.source_code, fname)

    _log(
        f"📊 {fname}(): CC={static_metrics.cyclomatic_complexity}, "
        f"MI={static_metrics.maintainability_index:.1f}, "
        f"bytecode={bc_count} instructions"
    )

    return {
        "function_name": fname,
        "file_path": target.file_path,
        "static_metrics": {
            "cyclomatic_complexity": static_metrics.cyclomatic_complexity,
            "cognitive_complexity": static_metrics.cognitive_complexity,
            "maintainability_index": round(static_metrics.maintainability_index, 2),
            "halstead_volume": round(static_metrics.halstead_volume, 2),
            "halstead_difficulty": round(static_metrics.halstead_difficulty, 2),
            "halstead_effort": round(static_metrics.halstead_effort, 2),
            "halstead_bugs": round(static_metrics.halstead_bugs, 4),
            "loc": static_metrics.loc,
            "nesting_depth": static_metrics.nesting_depth,
            "parameter_count": static_metrics.parameter_count,
        },
        "bytecode": {
            "instruction_count": bc_count,
            "categories": bc_categories,
        },
    }