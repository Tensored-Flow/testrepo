"""Deterministic metrics engine. No LLM calls. No randomness. Same input = same output.

Exports:
    compute_static_metrics(source_code, function_name) -> dict
    compute_cognitive_complexity(source_code) -> int
    benchmark_function(func, input_sizes, input_generator, iterations) -> list[dict]
    estimate_big_o(benchmarks) -> dict
    compute_rubric(before_metrics, after_metrics, before_benchmarks, after_benchmarks, test_results) -> dict
"""

import ast
import copy
import json
import random
import time
import tracemalloc

import lizard
import numpy as np
from radon.metrics import h_visit, mi_visit


# ════════════════════════════════════════════════════════════
# FUNCTION 1: compute_static_metrics
# ════════════════════════════════════════════════════════════

def compute_static_metrics(source_code: str, function_name: str) -> dict:
    """Run lizard + radon on source code. Returns all static analysis numbers."""

    # --- Lizard: McCabe cyclomatic complexity ---
    analysis = lizard.analyze_file.analyze_source_code("file.py", source_code)

    target = None
    for func in analysis.function_list:
        if func.name == function_name:
            target = func
            break

    # Substring fallback (lizard may prefix with class name)
    if target is None:
        for func in analysis.function_list:
            if function_name in func.name:
                target = func
                break

    # Last resort: first function or aggregate
    if target is None and analysis.function_list:
        target = analysis.function_list[0]

    cc = target.cyclomatic_complexity if target else 1
    nloc = target.nloc if target else 0
    nesting = target.max_nesting_depth if target else 0
    param_count = len(target.parameters) if target else 0

    # --- Radon: Halstead metrics ---
    try:
        h_results = h_visit(source_code)
        # h_visit returns Halstead(total=HalsteadReport, functions=[(name, HalsteadReport)])
        if hasattr(h_results, "total"):
            h = h_results.total
        elif isinstance(h_results, list) and len(h_results) > 0:
            h = h_results[0]
        else:
            h = h_results
    except Exception:
        h = None

    halstead_volume = round(h.volume, 2) if h else None
    halstead_difficulty = round(h.difficulty, 2) if h else None
    halstead_bugs = round(h.bugs, 3) if h else None

    # --- Radon: Maintainability Index ---
    try:
        mi_result = mi_visit(source_code, False)
        if isinstance(mi_result, (list, tuple)):
            mi = float(mi_result[0]) if mi_result else 50.0
        else:
            mi = float(mi_result)
        mi = round(mi, 1)
    except Exception:
        mi = 50.0

    # --- Cognitive complexity ---
    cog = compute_cognitive_complexity(source_code)

    return {
        "cyclomatic_complexity": cc,
        "cognitive_complexity": cog,
        "maintainability_index": mi,
        "halstead_volume": halstead_volume,
        "halstead_difficulty": halstead_difficulty,
        "halstead_estimated_bugs": halstead_bugs,
        "nesting_depth": nesting,
        "nloc": nloc,
        "parameter_count": param_count,
    }


# ════════════════════════════════════════════════════════════
# FUNCTION 2: compute_cognitive_complexity
# ════════════════════════════════════════════════════════════

def compute_cognitive_complexity(source_code: str) -> int:
    """SonarSource cognitive complexity via AST walking."""
    tree = ast.parse(source_code)
    score = 0

    def visit_body(stmts, nesting):
        for stmt in stmts:
            visit_node(stmt, nesting)

    def visit_node(node, nesting):
        nonlocal score

        if isinstance(node, ast.If):
            handle_if_chain(node, nesting, is_elif=False)

        elif isinstance(node, (ast.For, ast.AsyncFor)):
            score += 1 + nesting
            count_boolops(node.iter)
            visit_body(node.body, nesting + 1)
            if node.orelse:
                score += 1
                visit_body(node.orelse, nesting + 1)

        elif isinstance(node, ast.While):
            score += 1 + nesting
            count_boolops(node.test)
            visit_body(node.body, nesting + 1)
            if node.orelse:
                score += 1
                visit_body(node.orelse, nesting + 1)

        elif isinstance(node, (ast.With, ast.AsyncWith)):
            score += 1 + nesting
            visit_body(node.body, nesting + 1)

        elif isinstance(node, ast.Try):
            visit_body(node.body, nesting)
            for handler in node.handlers:
                score += 1 + nesting
                visit_body(handler.body, nesting + 1)
            if node.orelse:
                visit_body(node.orelse, nesting)
            if hasattr(node, "finalbody") and node.finalbody:
                visit_body(node.finalbody, nesting)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Nested function definition increases nesting
            visit_body(node.body, nesting + 1)

        elif isinstance(node, ast.ClassDef):
            visit_body(node.body, nesting + 1)

        elif isinstance(node, (ast.Break, ast.Continue)):
            score += 1

        elif isinstance(node, ast.Expr):
            count_boolops(node.value)

        elif isinstance(node, (ast.Assign, ast.AugAssign)):
            if hasattr(node, "value") and node.value:
                count_boolops(node.value)

        elif isinstance(node, ast.Return):
            if node.value:
                count_boolops(node.value)

    def handle_if_chain(node, nesting, is_elif):
        nonlocal score

        if is_elif:
            score += 1                # +1 for elif, no nesting bonus
        else:
            score += 1 + nesting      # +1 + nesting for if

        count_boolops(node.test)
        visit_body(node.body, nesting + 1)

        if node.orelse:
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                # elif: recurse without increasing nesting
                handle_if_chain(node.orelse[0], nesting, is_elif=True)
            else:
                # else block
                score += 1
                visit_body(node.orelse, nesting + 1)

    def count_boolops(expr_node):
        """Count BoolOp nodes in an expression (each = one sequence of same operator)."""
        nonlocal score
        if expr_node is None:
            return
        for child in ast.walk(expr_node):
            if isinstance(child, ast.BoolOp):
                score += 1

    # Walk top-level function definitions at nesting 0
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            visit_body(node.body, 0)

    return score


# ════════════════════════════════════════════════════════════
# FUNCTION 3: benchmark_function
# ════════════════════════════════════════════════════════════

def benchmark_function(func, input_sizes=None, input_generator=None, iterations=10):
    """Empirically benchmark a callable at multiple input sizes."""
    if input_sizes is None:
        input_sizes = [100, 500, 1000, 5000, 10000]

    if input_generator is None:
        random.seed(42)

        def input_generator(size):
            return [random.randint(-10000, 10000) for _ in range(size)]

    results = []

    for size in input_sizes:
        input_data = input_generator(size)
        timed_out = False

        # Warmup run (untimed, primes caches)
        try:
            func(copy.deepcopy(input_data))
        except Exception:
            pass

        # Loop 1: Timing only (no tracemalloc overhead)
        times = []
        for _ in range(iterations):
            input_copy = copy.deepcopy(input_data)
            start = time.perf_counter()
            func(input_copy)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)

            if elapsed_ms > 10_000:  # 10-second timeout
                timed_out = True
                break

        # Loop 2: Memory measurement (separate, not timed)
        tracemalloc.start()
        func(copy.deepcopy(input_data))
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results.append({
            "input_size": size,
            "time_ms": round(min(times), 3),
            "memory_mb": round(max(peak_bytes, 0) / 1024 / 1024, 3),
        })

        if timed_out:
            break

    return results


# ════════════════════════════════════════════════════════════
# FUNCTION 4: estimate_big_o
# ════════════════════════════════════════════════════════════

def estimate_big_o(benchmarks: list) -> dict:
    """Fit Big O curve via log-log linear regression."""
    sizes = [b["input_size"] for b in benchmarks]
    times = [b["time_ms"] for b in benchmarks]

    valid = [(s, t) for s, t in zip(sizes, times) if t > 0 and s > 0]

    if len(valid) < 2:
        return {"label": "O(?)", "slope": 0.0, "r_squared": 0.0, "confidence": "low"}

    log_sizes = np.log(np.array([v[0] for v in valid]))
    log_times = np.log(np.array([v[1] for v in valid]))

    slope, intercept = np.polyfit(log_sizes, log_times, 1)
    slope = round(float(slope), 2)

    # R-squared
    predicted = slope * log_sizes + intercept
    ss_res = float(np.sum((log_times - predicted) ** 2))
    ss_tot = float(np.sum((log_times - np.mean(log_times)) ** 2))
    r_squared = round(1 - (ss_res / ss_tot), 3) if ss_tot > 0 else 0.0

    # Map slope to Big O label
    if slope < 0.1:
        label = "O(1)"
    elif slope < 0.7:
        label = "O(sqrt(n))"
    elif slope < 1.2:
        label = "O(n)"
    elif slope < 1.7:
        label = "O(n log n)"
    elif slope < 2.3:
        label = "O(n^2)"
    elif slope < 3.3:
        label = "O(n^3)"
    else:
        label = f"O(n^{slope})"

    if r_squared > 0.95:
        confidence = "high"
    elif r_squared > 0.8:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "label": label,
        "slope": slope,
        "r_squared": r_squared,
        "confidence": confidence,
    }


# ════════════════════════════════════════════════════════════
# FUNCTION 5: compute_rubric
# ════════════════════════════════════════════════════════════

def compute_rubric(before_metrics, after_metrics,
                   before_benchmarks, after_benchmarks,
                   test_results):
    """Deterministic pass/fail. No AI judgment. Pure math."""

    vetoes = []

    # ── Hard vetoes ──
    if not test_results["existing_tests_passed"]:
        vetoes.append("Existing tests failed")

    if not test_results["differential_tests_passed"]:
        failures = test_results.get("differential_failures", [])
        vetoes.append(f"Differential tests failed: {', '.join(failures) if failures else 'unknown'}")

    if not test_results["targeted_tests_passed"]:
        failures = test_results.get("targeted_failures", [])
        vetoes.append(f"Targeted tests failed: {', '.join(failures) if failures else 'unknown'}")

    if before_benchmarks and after_benchmarks:
        avg_mem_before = sum(b["memory_mb"] for b in before_benchmarks) / len(before_benchmarks)
        avg_mem_after = sum(b["memory_mb"] for b in after_benchmarks) / len(after_benchmarks)
        mem_diff = avg_mem_after - avg_mem_before
        if avg_mem_before > 0 and avg_mem_after > avg_mem_before * 1.20 and mem_diff > 0.1:
            vetoes.append(f"Memory regression: {avg_mem_before:.3f}MB → {avg_mem_after:.3f}MB (>20% increase)")

        avg_time_before = sum(b["time_ms"] for b in before_benchmarks) / len(before_benchmarks)
        avg_time_after = sum(b["time_ms"] for b in after_benchmarks) / len(after_benchmarks)
        if avg_time_before > 0 and avg_time_after > avg_time_before * 1.10:
            vetoes.append(f"Runtime regression: {avg_time_before:.3f}ms → {avg_time_after:.3f}ms (>10% increase)")

    if vetoes:
        return {
            "verdict": "REJECTED",
            "vetoes": vetoes,
            "improvements": [],
            "rejection_reason": vetoes[0],
        }

    # ── Improvement checks ──
    improvements = []

    if after_metrics["cyclomatic_complexity"] < before_metrics["cyclomatic_complexity"]:
        improvements.append(
            f"Cyclomatic complexity: {before_metrics['cyclomatic_complexity']} → {after_metrics['cyclomatic_complexity']}"
        )

    if after_metrics["cognitive_complexity"] < before_metrics["cognitive_complexity"]:
        improvements.append(
            f"Cognitive complexity: {before_metrics['cognitive_complexity']} → {after_metrics['cognitive_complexity']}"
        )

    if after_metrics["maintainability_index"] > before_metrics["maintainability_index"]:
        improvements.append(
            f"Maintainability index: {before_metrics['maintainability_index']} → {after_metrics['maintainability_index']}"
        )

    if (before_metrics["halstead_difficulty"] is not None
            and after_metrics["halstead_difficulty"] is not None
            and after_metrics["halstead_difficulty"] < before_metrics["halstead_difficulty"]):
        improvements.append(
            f"Halstead difficulty: {before_metrics['halstead_difficulty']} → {after_metrics['halstead_difficulty']}"
        )

    if (before_metrics["halstead_estimated_bugs"] is not None
            and after_metrics["halstead_estimated_bugs"] is not None
            and after_metrics["halstead_estimated_bugs"] < before_metrics["halstead_estimated_bugs"]):
        improvements.append(
            f"Halstead estimated bugs: {before_metrics['halstead_estimated_bugs']} → {after_metrics['halstead_estimated_bugs']}"
        )

    if before_benchmarks and after_benchmarks:
        avg_time_before = sum(b["time_ms"] for b in before_benchmarks) / len(before_benchmarks)
        avg_time_after = sum(b["time_ms"] for b in after_benchmarks) / len(after_benchmarks)
        if avg_time_before > 0 and avg_time_after < avg_time_before * 0.90:
            improvements.append(f"Runtime: {avg_time_before:.3f}ms → {avg_time_after:.3f}ms")

    if before_benchmarks and after_benchmarks:
        avg_mem_before = sum(b["memory_mb"] for b in before_benchmarks) / len(before_benchmarks)
        avg_mem_after = sum(b["memory_mb"] for b in after_benchmarks) / len(after_benchmarks)
        if avg_mem_before > 0 and avg_mem_after < avg_mem_before * 0.90:
            improvements.append(f"Memory: {avg_mem_before:.3f}MB → {avg_mem_after:.3f}MB")

    if before_benchmarks and after_benchmarks:
        before_big_o = estimate_big_o(before_benchmarks)
        after_big_o = estimate_big_o(after_benchmarks)
        if before_big_o["slope"] - after_big_o["slope"] > 0.3:
            improvements.append(
                f"Big O: {before_big_o['label']} (slope {before_big_o['slope']}) → {after_big_o['label']} (slope {after_big_o['slope']})"
            )

    if not improvements:
        return {
            "verdict": "REJECTED",
            "vetoes": [],
            "improvements": [],
            "rejection_reason": "No meaningful improvement detected",
        }

    return {
        "verdict": "APPROVED",
        "vetoes": [],
        "improvements": improvements,
        "rejection_reason": None,
    }
