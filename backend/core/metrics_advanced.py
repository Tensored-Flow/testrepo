"""Advanced metrics: bytecode analysis and statistical significance testing.

Sits ON TOP of the basic metrics engine. Does NOT duplicate any basic metrics.
These two functions give technical depth beyond what source-level tools provide.

Exports:
    analyze_bytecode(source_code, function_name) -> dict
    benchmark_with_significance(original_func, optimized_func, ...) -> dict
    compare_bytecode(original_source, optimized_source, function_name) -> dict
"""

from __future__ import annotations

import copy
import dis
import random
import time
from typing import Any, Callable

import numpy as np
from scipy import stats


# ════════════════════════════════════════════════════════════
# FUNCTION 1: analyze_bytecode
# ════════════════════════════════════════════════════════════

def analyze_bytecode(source_code: str, function_name: str) -> dict[str, Any]:
    """Analyze CPython bytecode instructions for a function.

    Goes a level BELOW source code — actual VM instructions the
    interpreter will execute. Categorizes by operation type.
    """
    try:
        namespace = {}
        exec(source_code, namespace)
    except Exception as e:
        return {"error": f"exec failed: {e}"}

    func = namespace.get(function_name)
    if func is None:
        # Try to find it (might be nested or slightly different name)
        for name, obj in namespace.items():
            if callable(obj) and function_name in name:
                func = obj
                break
    if func is None:
        return {"error": f"function '{function_name}' not found"}

    code_obj = func.__code__
    instructions = list(dis.get_instructions(code_obj))

    total_instructions = len(instructions)

    categories = {
        "load_store": 0,
        "binary_ops": 0,
        "control_flow": 0,
        "function_calls": 0,
        "stack_ops": 0,
        "build_ops": 0,
        "other": 0,
    }

    for instr in instructions:
        opname = instr.opname
        if "LOAD" in opname or "STORE" in opname:
            categories["load_store"] += 1
        elif "BINARY" in opname or "COMPARE" in opname or "UNARY" in opname:
            categories["binary_ops"] += 1
        elif "JUMP" in opname or "FOR_ITER" in opname or "SETUP" in opname:
            categories["control_flow"] += 1
        elif "CALL" in opname:
            categories["function_calls"] += 1
        elif "POP" in opname or "DUP" in opname or "ROT" in opname or "COPY" in opname:
            categories["stack_ops"] += 1
        elif "BUILD" in opname or "LIST" in opname or "MAP" in opname:
            categories["build_ops"] += 1
        else:
            categories["other"] += 1

    unique_opcodes = len(set(i.opname for i in instructions))

    loop_count = sum(
        1 for i in instructions
        if "FOR_ITER" in i.opname or i.opname == "GET_ITER"
    )

    source_lines = len([
        l for l in source_code.strip().split("\n")
        if l.strip() and not l.strip().startswith("#")
    ])
    instruction_density = round(total_instructions / max(source_lines, 1), 2)

    return {
        "total_instructions": total_instructions,
        "categories": categories,
        "unique_opcodes": unique_opcodes,
        "loop_instructions": loop_count,
        "instruction_density": instruction_density,
        "summary": (
            f"{total_instructions} bytecode ops, "
            f"{categories['control_flow']} control flow, "
            f"{categories['function_calls']} calls, "
            f"density {instruction_density} ops/line"
        ),
    }


# ════════════════════════════════════════════════════════════
# FUNCTION 2: benchmark_with_significance
# ════════════════════════════════════════════════════════════

def benchmark_with_significance(
    original_func: Callable,
    optimized_func: Callable,
    input_sizes: list[int] | None = None,
    input_generator: Callable[[int], Any] | None = None,
    runs_per_size: int = 10,
) -> dict[str, Any]:
    """Run statistically rigorous benchmarks with Welch's t-test.

    Proves improvement isn't just noise: p < 0.05 or it didn't happen.
    """
    if input_sizes is None:
        input_sizes = [500, 1000, 5000]

    if input_generator is None:
        random.seed(42)

        def input_generator(size):
            return [random.randint(-10000, 10000) for _ in range(size)]

    runs_per_size = max(runs_per_size, 5)  # minimum for statistical validity

    results_per_size = []

    for size in input_sizes:
        input_data = input_generator(size)

        original_times = []
        optimized_times = []
        timed_out = False

        for _ in range(runs_per_size):
            # Time original
            inp = copy.deepcopy(input_data)
            start = time.perf_counter()
            original_func(inp)
            elapsed = (time.perf_counter() - start) * 1000
            original_times.append(elapsed)

            if elapsed > 10_000:
                timed_out = True
                break

            # Time optimized
            inp = copy.deepcopy(input_data)
            start = time.perf_counter()
            try:
                optimized_func(inp)
            except Exception as e:
                results_per_size.append({
                    "input_size": size,
                    "error": f"optimized_func threw: {e}",
                })
                continue
            optimized_times.append((time.perf_counter() - start) * 1000)

        if timed_out or len(original_times) < 5 or len(optimized_times) < 5:
            results_per_size.append({
                "input_size": size,
                "error": "timed out or insufficient samples",
                "original_mean_ms": round(np.mean(original_times), 3) if original_times else 0,
                "optimized_mean_ms": round(np.mean(optimized_times), 3) if optimized_times else 0,
                "significant": False,
            })
            if timed_out:
                break
            continue

        orig_arr = np.array(original_times)
        opt_arr = np.array(optimized_times)

        # Welch's t-test (doesn't assume equal variance)
        if orig_arr.std() == 0 and opt_arr.std() == 0:
            # No variance — can't run t-test
            p_value = 0.0 if orig_arr.mean() != opt_arr.mean() else 1.0
            t_stat = 0.0
        else:
            t_stat, p_value = stats.ttest_ind(orig_arr, opt_arr, equal_var=False)

        # Cohen's d effect size
        pooled_std = np.sqrt((orig_arr.std() ** 2 + opt_arr.std() ** 2) / 2)
        cohens_d = float((orig_arr.mean() - opt_arr.mean()) / pooled_std) if pooled_std > 0 else 0.0

        # Speedup ratio
        speedup = float(orig_arr.mean() / opt_arr.mean()) if opt_arr.mean() > 0 else float("inf")

        # 95% confidence interval for the time difference
        diff = orig_arr[:len(opt_arr)] - opt_arr[:len(orig_arr)]
        if len(diff) >= 2 and np.std(diff) > 0:
            ci_low, ci_high = stats.t.interval(
                0.95, len(diff) - 1,
                loc=np.mean(diff),
                scale=stats.sem(diff),
            )
        else:
            ci_low, ci_high = 0.0, 0.0

        results_per_size.append({
            "input_size": size,
            "original_mean_ms": round(float(orig_arr.mean()), 3),
            "original_std_ms": round(float(orig_arr.std()), 3),
            "optimized_mean_ms": round(float(opt_arr.mean()), 3),
            "optimized_std_ms": round(float(opt_arr.std()), 3),
            "speedup": round(speedup, 2),
            "p_value": round(float(p_value), 6),
            "significant": float(p_value) < 0.05,
            "cohens_d": round(cohens_d, 3),
            "effect_size": (
                "large" if abs(cohens_d) > 0.8 else
                "medium" if abs(cohens_d) > 0.5 else
                "small" if abs(cohens_d) > 0.2 else
                "negligible"
            ),
            "ci_95_low_ms": round(float(ci_low), 3),
            "ci_95_high_ms": round(float(ci_high), 3),
        })

    # Overall verdict
    valid_results = [r for r in results_per_size if "error" not in r]
    all_significant = all(r["significant"] for r in valid_results) if valid_results else False
    avg_speedup = float(np.mean([r["speedup"] for r in valid_results])) if valid_results else 0.0

    return {
        "per_size": results_per_size,
        "overall_significant": all_significant,
        "average_speedup": round(avg_speedup, 2),
        "verdict": (
            f"Statistically significant improvement (p < 0.05 at all sizes, "
            f"{avg_speedup:.1f}x average speedup)"
            if all_significant else
            "Improvement not statistically significant at all input sizes"
        ),
    }


# ════════════════════════════════════════════════════════════
# FUNCTION 3 (BONUS): compare_bytecode
# ════════════════════════════════════════════════════════════

def compare_bytecode(
    original_source: str,
    optimized_source: str,
    function_name: str,
) -> dict[str, Any]:
    """Run analyze_bytecode on both versions and produce a comparison."""
    before = analyze_bytecode(original_source, function_name)
    after = analyze_bytecode(optimized_source, function_name)

    if "error" in before or "error" in after:
        return {
            "error": before.get("error") or after.get("error"),
            "before": before,
            "after": after,
        }

    return {
        "total_instructions": {
            "before": before["total_instructions"],
            "after": after["total_instructions"],
            "delta": after["total_instructions"] - before["total_instructions"],
            "improved": after["total_instructions"] < before["total_instructions"],
        },
        "control_flow": {
            "before": before["categories"]["control_flow"],
            "after": after["categories"]["control_flow"],
            "delta": after["categories"]["control_flow"] - before["categories"]["control_flow"],
            "improved": after["categories"]["control_flow"] < before["categories"]["control_flow"],
        },
        "instruction_density": {
            "before": before["instruction_density"],
            "after": after["instruction_density"],
            "delta": round(after["instruction_density"] - before["instruction_density"], 2),
            "improved": after["instruction_density"] < before["instruction_density"],
        },
        "summary": (
            f"Bytecode: {before['total_instructions']} → {after['total_instructions']} instructions "
            f"({before['categories']['control_flow']} → {after['categories']['control_flow']} control flow ops)"
        ),
    }
