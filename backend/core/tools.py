"""
Deterministic Tool Definitions + Execution Handlers.

This is the bridge between Claude's reasoning and our math.
Each tool has:
  1. A SCHEMA (JSON) that Claude sees and decides whether to call
  2. A HANDLER (Python) that executes deterministically

Claude as the Analyst/Optimizer AUTONOMOUSLY decides which tools to call,
interprets the results, and reasons about next steps. That's what makes
this agentic — the AI chooses its own investigation path.

"The AI proposes, the math disposes" — but now the AI also decides
WHICH math to run.
"""

import json
import ast
import textwrap
import difflib
import dis as dis_module
import io
from typing import Any, Callable

# ─── Lazy imports (installed at agent startup) ───
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



# ═══════════════════════════════════════════════════════════
# TOOL SCHEMAS — What Claude sees
# ═══════════════════════════════════════════════════════════

# ─── Analyst Tools (investigation) ───

RUN_LIZARD_TOOL = {
    "name": "run_lizard",
    "description": (
        "Compute cyclomatic complexity (McCabe's CC = E - N + 2P) for a Python function. "
        "Returns CC score, number of lines, token count, and parameter count. "
        "CC > 10 means high complexity, CC > 20 means very high."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "source_code": {
                "type": "string",
                "description": "The complete Python function source code"
            },
        },
        "required": ["source_code"]
    }
}

RUN_RADON_TOOL = {
    "name": "run_radon",
    "description": (
        "Compute Halstead metrics and Maintainability Index for a Python function. "
        "Returns: volume (program size), difficulty (error-proneness), effort (mental effort to understand), "
        "estimated bugs (V/3000), and MI (171 - 5.2×ln(V) - 0.23×CC - 16.2×ln(LOC)). "
        "MI > 20 is good, MI < 10 is very hard to maintain."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "source_code": {
                "type": "string",
                "description": "The complete Python function source code"
            },
        },
        "required": ["source_code"]
    }
}

ANALYZE_BYTECODE_TOOL = {
    "name": "analyze_bytecode",
    "description": (
        "Disassemble a Python function to CPython bytecode using the dis module. "
        "Returns: total instruction count, instructions grouped by category "
        "(memory ops, calls, branches, comparisons, arithmetic, iteration, data structures). "
        "Heavy 'iteration' + 'branches' suggests loop-heavy code. Heavy 'calls' suggests function call overhead."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "source_code": {
                "type": "string",
                "description": "Complete Python function source code"
            },
            "function_name": {
                "type": "string",
                "description": "Name of the function to disassemble"
            },
        },
        "required": ["source_code", "function_name"]
    }
}

BENCHMARK_FUNCTION_TOOL = {
    "name": "benchmark_function",
    "description": (
        "Run timing and memory benchmarks on a Python function at multiple input sizes. "
        "Uses time.perf_counter (nanosecond precision) and tracemalloc (byte-level memory). "
        "Returns mean time, std deviation, and peak memory for each size. "
        "Use this to empirically determine the function's growth rate."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "source_code": {
                "type": "string",
                "description": "Complete Python function source code"
            },
            "function_name": {
                "type": "string",
                "description": "Name of the function"
            },
            "input_generator": {
                "type": "string",
                "description": (
                    "Python code that defines generate_input(n) returning a tuple of args. "
                    "Example: 'import random\\ndef generate_input(n):\\n    return ([random.randint(0,n*10) for _ in range(n)],)'"
                )
            },
            "sizes": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Input sizes to benchmark at. Default: [100, 500, 1000, 5000]"
            },
        },
        "required": ["source_code", "function_name", "input_generator"]
    }
}

ESTIMATE_BIG_O_TOOL = {
    "name": "estimate_big_o",
    "description": (
        "Estimate Big O complexity from benchmark timing data using log-log linear regression. "
        "Fits log(time) = slope × log(n) + intercept. Slope ≈1 means O(n), ≈2 means O(n²), etc. "
        "Reports R² as a confidence metric for the fit quality."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sizes": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Input sizes"
            },
            "times": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Mean execution times corresponding to each size"
            },
        },
        "required": ["sizes", "times"]
    }
}

RUN_FUNCTION_WITH_INPUTS_TOOL = {
    "name": "run_function_with_inputs",
    "description": (
        "Execute a Python function with specific test inputs and capture outputs. "
        "Use this to understand what the function does, test edge cases, or compare original vs optimized behavior. "
        "Returns output value, output type, and any exceptions for each input."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "source_code": {
                "type": "string",
                "description": "Complete Python function source code"
            },
            "function_name": {
                "type": "string",
                "description": "Name of the function"
            },
            "test_inputs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "args": {"type": "array"},
                        "kwargs": {"type": "object"}
                    }
                },
                "description": "List of {args: [...], kwargs: {...}} to test"
            },
        },
        "required": ["source_code", "function_name", "test_inputs"]
    }
}

# ─── NEW: Web Research Tool ───

WEB_RESEARCH_TOOL = {
    "name": "web_research",
    "description": (
        "Search the web for information about algorithms, libraries, optimization techniques, or best practices. "
        "Use this when you encounter an unfamiliar library, need to understand the optimal algorithm for a problem, "
        "or want to check if a standard library function exists that replaces custom code. Returns relevant findings."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Search query. Be specific. Examples: 'python fastest way to deduplicate list', "
                    "'numpy vectorized alternative to nested loop', 'time complexity of heapq.nsmallest'"
                )
            },
            "context": {
                "type": "string",
                "description": "Brief context about why you're researching this (helps focus results)"
            }
        },
        "required": ["query"]
    }
}

# ─── NEW: Pattern Lookup Tool ───

PATTERN_LOOKUP_TOOL = {
    "name": "lookup_optimization_pattern",
    "description": (
        "Check if the code matches a known optimization pattern from our database. "
        "Use this FIRST before doing heavy analysis — it might be a well-known antipattern with a standard fix."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "pattern_description": {
                "type": "string",
                "description": (
                    "Describe the code pattern you see. E.g. 'nested loop with list.index() for membership check', "
                    "'manual implementation of sorting', 'repeated string concatenation in loop'"
                )
            },
            "language_constructs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key constructs used. E.g. ['for loop', 'list.append', 'string concatenation']"
            }
        },
        "required": ["pattern_description"]
    }
}

# ─── NEW: Diff Analysis Tool ───

DIFF_ANALYSIS_TOOL = {
    "name": "analyze_code_diff",
    "description": (
        "Compare original and optimized code side-by-side and identify exactly what changed. "
        "Use this during rejection diagnosis to understand what the Optimizer actually did vs what was intended."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "original_code": {"type": "string"},
            "optimized_code": {"type": "string"}
        },
        "required": ["original_code", "optimized_code"]
    }
}

# ─── NEW: Metric Comparison Tool ───

METRIC_COMPARISON_TOOL = {
    "name": "compare_metrics_detailed",
    "description": (
        "Run full metrics on both original and optimized code and produce a side-by-side comparison table. "
        "Use this to understand exactly which metrics improved, regressed, or stayed the same."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "original_code": {"type": "string"},
            "optimized_code": {"type": "string"},
            "function_name": {"type": "string"}
        },
        "required": ["original_code", "optimized_code", "function_name"]
    }
}

# ─── NEW: Failure Hypothesis Tool ───

FAILURE_HYPOTHESIS_TOOL = {
    "name": "generate_failure_hypotheses",
    "description": (
        "Given the rejection details (which vetoes fired, which metrics regressed), generate ranked hypotheses "
        "about WHY the optimization failed. Use this to structure your diagnosis on Round 2+."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "vetoes_triggered": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of veto names that fired (from Validator)"
            },
            "metric_details": {
                "type": "object",
                "description": "Before/after metric values from Validator"
            },
            "hypothesis_from_previous_round": {
                "type": "string",
                "description": "What the Analyst hypothesized last round"
            }
        },
        "required": ["vetoes_triggered"]
    }
}

# ─── Optimizer Tools (validation) ───

COMPILE_CHECK_TOOL = {
    "name": "compile_check",
    "description": (
        "Check if Python code is syntactically valid and the target function can be extracted. "
        "Returns: valid (bool), error message if invalid, function signature if valid."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "source_code": {
                "type": "string",
                "description": "Python code to validate"
            },
            "function_name": {
                "type": "string",
                "description": "Expected function name"
            },
        },
        "required": ["source_code", "function_name"]
    }
}

COMPARE_OUTPUTS_TOOL = {
    "name": "compare_outputs",
    "description": (
        "Run both original and optimized functions with the same inputs and compare outputs. "
        "Returns whether all outputs match, plus details of any mismatches. "
        "This is a quick behavioral equivalence check before full validation."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "original_source": {
                "type": "string",
                "description": "Original function source code"
            },
            "optimized_source": {
                "type": "string",
                "description": "Optimized function source code"
            },
            "function_name": {
                "type": "string",
                "description": "Function name (same in both)"
            },
            "test_inputs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "args": {"type": "array"},
                        "kwargs": {"type": "object"}
                    }
                },
                "description": "Inputs to test both functions with"
            },
        },
        "required": ["original_source", "optimized_source", "function_name", "test_inputs"]
    }
}

RUN_TESTS_TOOL = {
    "name": "run_tests",
    "description": (
        "Execute Python test code and return pass/fail results. "
        "The test code should use assert statements and print 'ALL TESTS PASSED' on success."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "test_code": {
                "type": "string",
                "description": "Python test code to execute"
            },
        },
        "required": ["test_code"]
    }
}

QUICK_BENCHMARK_TOOL = {
    "name": "quick_benchmark",
    "description": (
        "Quick benchmark comparing original vs optimized function performance. "
        "Returns speedup ratio and whether the optimization is faster."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "original_source": {
                "type": "string",
                "description": "Original function source"
            },
            "optimized_source": {
                "type": "string",
                "description": "Optimized function source"
            },
            "function_name": {
                "type": "string",
                "description": "Function name"
            },
            "input_generator": {
                "type": "string",
                "description": "Python code defining generate_input(n)"
            },
            "size": {
                "type": "integer",
                "description": "Input size to benchmark at. Default: 1000"
            },
        },
        "required": ["original_source", "optimized_source", "function_name", "input_generator"]
    }
}


# ═══════════════════════════════════════════════════════════
# Tool registries per agent role
# ═══════════════════════════════════════════════════════════

ANALYST_TOOLS = [
    RUN_LIZARD_TOOL,
    RUN_RADON_TOOL,
    ANALYZE_BYTECODE_TOOL,
    BENCHMARK_FUNCTION_TOOL,
    ESTIMATE_BIG_O_TOOL,
    RUN_FUNCTION_WITH_INPUTS_TOOL,
    WEB_RESEARCH_TOOL,
    PATTERN_LOOKUP_TOOL,
    DIFF_ANALYSIS_TOOL,
    METRIC_COMPARISON_TOOL,
    FAILURE_HYPOTHESIS_TOOL,
]

OPTIMIZER_TOOLS = [
    COMPILE_CHECK_TOOL,
    COMPARE_OUTPUTS_TOOL,
    RUN_TESTS_TOOL,
    QUICK_BENCHMARK_TOOL,
    WEB_RESEARCH_TOOL,
]

TEST_DESIGNER_TOOLS = [
    RUN_TESTS_TOOL,
    RUN_FUNCTION_WITH_INPUTS_TOOL,
    COMPARE_OUTPUTS_TOOL,
]

# Combine for agents that need all tools
ALL_TOOLS = ANALYST_TOOLS + OPTIMIZER_TOOLS


# ═══════════════════════════════════════════════════════════
# TOOL HANDLERS — What actually executes (deterministic)
# ═══════════════════════════════════════════════════════════

def handle_run_lizard(source_code: str, **kwargs) -> dict:
    """Execute lizard cyclomatic complexity analysis."""
    if not lizard:
        return {"error": "lizard not installed"}

    try:
        analysis = lizard.analyze_file.analyze_source_code("<input>", source_code)
        if analysis.function_list:
            func = analysis.function_list[0]
            return {
                "cyclomatic_complexity": func.cyclomatic_complexity,
                "lines_of_code": func.nloc,
                "token_count": func.token_count,
                "parameter_count": func.length,
                "function_name": func.name,
            }
        return {"cyclomatic_complexity": 1, "note": "No functions found in source"}
    except Exception as e:
        return {"error": str(e)}


def handle_run_radon(source_code: str, **kwargs) -> dict:
    """Execute radon Halstead + Maintainability Index analysis."""
    result = {
        "halstead_volume": 0, "halstead_difficulty": 0,
        "halstead_effort": 0, "halstead_bugs": 0,
        "maintainability_index": 0,
    }

    if not radon:
        result["error"] = "radon not installed"
        return result

    try:
        halstead = radon.metrics.h_visit(source_code)
        if halstead:
            h = halstead[0] if isinstance(halstead, list) else halstead
            result["halstead_volume"] = round(getattr(h, 'volume', 0) or 0, 2)
            result["halstead_difficulty"] = round(getattr(h, 'difficulty', 0) or 0, 2)
            result["halstead_effort"] = round(getattr(h, 'effort', 0) or 0, 2)
            result["halstead_bugs"] = round(getattr(h, 'bugs', 0) or 0, 4)
    except Exception:
        pass

    try:
        mi = radon.metrics.mi_visit(source_code, True)
        result["maintainability_index"] = round(mi if isinstance(mi, (int, float)) else 0, 2)
    except Exception:
        pass

    return result


def handle_analyze_bytecode(source_code: str, function_name: str, **kwargs) -> dict:
    """Disassemble to CPython bytecode and categorize instructions."""
    try:
        namespace = {}
        exec(compile(textwrap.dedent(source_code), "<input>", "exec"), namespace)
        func = namespace.get(function_name)
        if not callable(func):
            return {"error": f"Function '{function_name}' not found or not callable"}

        instructions = list(dis_module.get_instructions(func))
        total = len(instructions)

        categories = {}
        for instr in instructions:
            cat = _categorize_opcode(instr.opname)
            categories[cat] = categories.get(cat, 0) + 1

        # Also get the raw disassembly for Claude to read
        output = io.StringIO()
        dis_module.dis(func, file=output)
        raw_disasm = output.getvalue()

        return {
            "total_instructions": total,
            "categories": categories,
            "top_instructions": _top_opcodes(instructions, 5),
            "disassembly_preview": raw_disasm[:1000],  # First 1000 chars
        }
    except Exception as e:
        return {"error": str(e)}


def handle_benchmark_function(
    source_code: str, function_name: str, input_generator: str,
    sizes: list = None, **kwargs
) -> dict:
    """Benchmark function at multiple input sizes."""
    from backend.core.sandbox import benchmark_function as sb_benchmark

    if sizes is None:
        sizes = [100, 500, 1000, 5000]

    results = sb_benchmark(source_code, function_name, input_generator, sizes=sizes)

    return {
        "benchmarks": results,
        "sizes_tested": sizes,
    }


def handle_estimate_big_o(sizes: list, times: list, **kwargs) -> dict:
    """Estimate Big O from timing data using log-log regression."""
    valid = [(s, t) for s, t in zip(sizes, times) if s > 0 and t > 0]
    if len(valid) < 2:
        return {"big_o": "unknown", "slope": 0, "confidence": "insufficient data"}

    log_sizes = [__import__('math').log(s) for s, _ in valid]
    log_times = [__import__('math').log(t) for _, t in valid]

    if np is not None:
        coeffs = np.polyfit(log_sizes, log_times, 1)
        slope = float(coeffs[0])
        # R² for confidence
        predicted = np.polyval(coeffs, log_sizes)
        ss_res = np.sum((np.array(log_times) - predicted) ** 2)
        ss_tot = np.sum((np.array(log_times) - np.mean(log_times)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    else:
        import math
        n = len(valid)
        sum_x = sum(log_sizes)
        sum_y = sum(log_times)
        sum_xy = sum(x * y for x, y in zip(log_sizes, log_times))
        sum_x2 = sum(x * x for x in log_sizes)
        denom = n * sum_x2 - sum_x ** 2
        slope = (n * sum_xy - sum_x * sum_y) / denom if denom else 0
        r_squared = 0

    big_o = _slope_to_big_o(slope)

    # Confidence based on R² of the log-log fit
    confidence = "high" if r_squared > 0.95 else "medium" if r_squared > 0.8 else "low"

    return {
        "big_o": big_o,
        "slope": round(slope, 3),
        "r_squared": round(r_squared, 4) if r_squared else None,
        "confidence": confidence,
    }


def handle_run_function_with_inputs(
    source_code: str, function_name: str, test_inputs: list, **kwargs
) -> dict:
    """Run function with specific inputs, capture outputs."""
    from backend.core.sandbox import run_function_with_inputs as sb_run

    results = sb_run(source_code, function_name, test_inputs)
    return {"results": results, "total_tests": len(test_inputs)}


def handle_compile_check(source_code: str, function_name: str, **kwargs) -> dict:
    """Validate Python syntax and function extraction."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        return {"valid": False, "error": f"Syntax error: {e}"}

    # Find the function
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == function_name:
                params = [arg.arg for arg in node.args.args]
                return {
                    "valid": True,
                    "function_name": function_name,
                    "parameters": params,
                    "line_count": node.end_lineno - node.lineno + 1 if node.end_lineno else 0,
                }

    return {"valid": False, "error": f"Function '{function_name}' not found"}


def handle_compare_outputs(
    original_source: str, optimized_source: str,
    function_name: str, test_inputs: list, **kwargs
) -> dict:
    """Compare original vs optimized outputs for same inputs."""
    from backend.core.sandbox import run_function_with_inputs as sb_run

    original_results = sb_run(original_source, function_name, test_inputs)
    optimized_results = sb_run(optimized_source, function_name, test_inputs)

    mismatches = []
    matches = 0

    for i, (orig, opt) in enumerate(zip(original_results, optimized_results)):
        orig_out = orig.get("output") if isinstance(orig, dict) else None
        opt_out = opt.get("output") if isinstance(opt, dict) else None
        orig_err = orig.get("error") if isinstance(orig, dict) else None
        opt_err = opt.get("error") if isinstance(opt, dict) else None

        if orig_out == opt_out and orig_err == opt_err:
            matches += 1
        else:
            mismatches.append({
                "input_index": i,
                "input": test_inputs[i] if i < len(test_inputs) else None,
                "original_output": orig_out,
                "optimized_output": opt_out,
                "original_error": orig_err,
                "optimized_error": opt_err,
            })

    return {
        "all_match": len(mismatches) == 0,
        "matches": matches,
        "mismatches": len(mismatches),
        "mismatch_details": mismatches[:5],  # Cap for readability
        "total_tests": len(test_inputs),
    }


def handle_run_tests(test_code: str, **kwargs) -> dict:
    """Execute test code in sandbox."""
    from backend.core.sandbox import run_in_sandbox

    result = run_in_sandbox(test_code, timeout=15.0)
    passed = "ALL TESTS PASSED" in (result.stdout or "") or "passed" in (result.stdout or "").lower()
    test_count = test_code.count("def test_") + test_code.count("assert ")

    return {
        "passed": passed,
        "success": result.success,
        "stdout": result.stdout[:2000] if result.stdout else "",
        "stderr": result.stderr[:1000] if result.stderr else "",
        "error": result.error,
        "estimated_assertions": test_count,
    }


def handle_quick_benchmark(
    original_source: str, optimized_source: str,
    function_name: str, input_generator: str,
    size: int = 1000, **kwargs
) -> dict:
    """Quick A/B benchmark of original vs optimized."""
    from backend.core.sandbox import benchmark_function as sb_benchmark

    orig_results = sb_benchmark(original_source, function_name, input_generator, sizes=[size], runs_per_size=10)
    opt_results = sb_benchmark(optimized_source, function_name, input_generator, sizes=[size], runs_per_size=10)

    orig_time = orig_results[0].get("mean_time", 0) if orig_results else 0
    opt_time = opt_results[0].get("mean_time", 0) if opt_results else 0

    speedup = orig_time / opt_time if opt_time > 0 else 0

    return {
        "original_time_ms": round(orig_time * 1000, 3),
        "optimized_time_ms": round(opt_time * 1000, 3),
        "speedup": round(speedup, 2),
        "faster": opt_time < orig_time,
        "improvement_percent": round((1 - opt_time / orig_time) * 100, 1) if orig_time > 0 else 0,
        "input_size": size,
    }


# ═══════════════════════════════════════════════════════════
# NEW TOOL HANDLERS
# ═══════════════════════════════════════════════════════════

async def handle_web_research(query: str, context: str = "", **kwargs) -> dict:
    """Execute web research for agents.

    Uses a nested Claude Haiku call with web_search tool for fast results.
    Falls back to returning a hint if the API doesn't support web_search.
    """
    try:
        import anthropic
        import os

        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{
                "role": "user",
                "content": (
                    f"Search for: {query}\nContext: {context}\n\n"
                    "Return a concise summary of the most relevant findings for code optimization purposes. "
                    "Focus on: time complexity, recommended algorithms, standard library alternatives, known pitfalls."
                )
            }]
        )
        text_parts = [block.text for block in response.content if hasattr(block, 'text')]
        return {
            "query": query,
            "findings": "\n".join(text_parts),
            "source": "web_research"
        }
    except Exception as e:
        return {
            "query": query,
            "findings": f"Web research unavailable: {e}. Proceeding with built-in knowledge.",
            "source": "fallback"
        }


# Known optimization patterns database
KNOWN_PATTERNS = {
    "nested_loop_membership": {
        "description": "Nested loop using `in` operator on list for membership checking",
        "antipattern": "O(n*m) list membership checks",
        "fix": "Convert to set for O(1) lookups: seen = set(items)",
        "complexity_change": "O(n*m) → O(n+m)",
        "confidence": 0.95
    },
    "bubble_sort": {
        "description": "Manual bubble sort or selection sort implementation",
        "antipattern": "O(n²) sorting",
        "fix": "Use Python's built-in sorted() or list.sort() which uses Timsort O(n log n)",
        "complexity_change": "O(n²) → O(n log n)",
        "confidence": 0.99
    },
    "string_concat_loop": {
        "description": "String concatenation using += in a loop",
        "antipattern": "O(n²) due to string immutability creating new objects each iteration",
        "fix": "Use list append + ''.join(), or io.StringIO",
        "complexity_change": "O(n²) → O(n)",
        "confidence": 0.9
    },
    "repeated_dict_access": {
        "description": "Accessing same dictionary key multiple times in a block",
        "antipattern": "Repeated hash computation",
        "fix": "Assign to local variable: val = d[key]; use val throughout",
        "complexity_change": "Constant factor improvement",
        "confidence": 0.7
    },
    "linear_search_sorted": {
        "description": "Linear search in sorted data or repeated linear searches",
        "antipattern": "O(n) search when O(log n) is possible",
        "fix": "Use bisect module for binary search on sorted data",
        "complexity_change": "O(n) → O(log n)",
        "confidence": 0.85
    },
    "manual_counting": {
        "description": "Manual counting with dict or defaultdict",
        "antipattern": "Manual counter implementation",
        "fix": "Use collections.Counter for counting",
        "complexity_change": "Same complexity but cleaner and slightly faster",
        "confidence": 0.8
    },
    "nested_comprehension_flat": {
        "description": "Nested list comprehension that could be itertools.chain",
        "antipattern": "Unnecessary intermediate lists",
        "fix": "Use itertools.chain.from_iterable() to avoid intermediate allocations",
        "complexity_change": "O(n) space → O(1) space (generator)",
        "confidence": 0.75
    },
    "repeated_max_min": {
        "description": "Multiple passes to find max, min, or other aggregates",
        "antipattern": "Multiple O(n) passes when one suffices",
        "fix": "Combine into single pass or use heapq.nlargest/nsmallest",
        "complexity_change": "k*O(n) → O(n)",
        "confidence": 0.8
    },
}


def handle_pattern_lookup(pattern_description: str, language_constructs: list = None, **kwargs) -> dict:
    """Match against known patterns using fuzzy keyword matching."""
    matches = []
    desc_lower = pattern_description.lower()
    for key, pattern in KNOWN_PATTERNS.items():
        keywords = key.split("_") + pattern["antipattern"].lower().split()
        score = sum(1 for kw in keywords if kw in desc_lower)
        if language_constructs:
            for construct in language_constructs:
                if construct.lower() in pattern["description"].lower():
                    score += 2
        if score >= 2:
            matches.append({**pattern, "match_key": key, "match_score": score})

    matches.sort(key=lambda m: m["match_score"], reverse=True)
    return {
        "matches_found": len(matches),
        "patterns": matches[:3],
        "note": "Use these as starting hypotheses. Still verify with tools."
    }


def handle_diff_analysis(original_code: str, optimized_code: str, **kwargs) -> dict:
    """Produce a structured diff with semantic annotations."""
    diff = list(difflib.unified_diff(
        original_code.splitlines(keepends=True),
        optimized_code.splitlines(keepends=True),
        fromfile="original", tofile="optimized"
    ))

    additions = sum(1 for l in diff if l.startswith('+') and not l.startswith('+++'))
    deletions = sum(1 for l in diff if l.startswith('-') and not l.startswith('---'))

    return {
        "diff_text": "".join(diff),
        "additions": additions,
        "deletions": deletions,
        "total_changes": additions + deletions,
        "change_ratio": round(
            (additions + deletions) / max(len(original_code.splitlines()), 1), 3
        ),
    }


def handle_metric_comparison(original_code: str, optimized_code: str, function_name: str, **kwargs) -> dict:
    """Run full metrics on both and produce side-by-side comparison."""
    orig_lizard = handle_run_lizard(original_code)
    opt_lizard = handle_run_lizard(optimized_code)
    orig_radon = handle_run_radon(original_code)
    opt_radon = handle_run_radon(optimized_code)

    comparisons = []
    metrics_to_compare = [
        ("Cyclomatic Complexity", orig_lizard.get("cyclomatic_complexity", 0), opt_lizard.get("cyclomatic_complexity", 0), False),
        ("Lines of Code", orig_lizard.get("lines_of_code", 0), opt_lizard.get("lines_of_code", 0), False),
        ("Halstead Volume", orig_radon.get("halstead_volume", 0), opt_radon.get("halstead_volume", 0), False),
        ("Halstead Difficulty", orig_radon.get("halstead_difficulty", 0), opt_radon.get("halstead_difficulty", 0), False),
        ("Halstead Bugs", orig_radon.get("halstead_bugs", 0), opt_radon.get("halstead_bugs", 0), False),
        ("Maintainability Index", orig_radon.get("maintainability_index", 0), opt_radon.get("maintainability_index", 0), True),
    ]

    for name, before, after, higher_is_better in metrics_to_compare:
        if higher_is_better:
            improved = after > before
        else:
            improved = after < before
        delta = ((after - before) / abs(before) * 100) if before else 0
        comparisons.append({
            "metric": name,
            "before": before,
            "after": after,
            "delta_percent": round(delta, 2),
            "improved": improved,
        })

    return {
        "function_name": function_name,
        "comparisons": comparisons,
        "overall_improved": any(c["improved"] for c in comparisons),
    }


def handle_failure_hypotheses(
    vetoes_triggered: list,
    metric_details: dict = None,
    hypothesis_from_previous_round: str = None,
    **kwargs,
) -> dict:
    """Generate ranked hypotheses about WHY the optimization failed."""
    hypotheses = []

    for veto in vetoes_triggered:
        veto_lower = veto.lower()
        if "differential" in veto_lower or "behavioral" in veto_lower:
            hypotheses.append({
                "hypothesis": "The optimization changed the function's behavior — it produces different outputs for some inputs.",
                "likely_cause": "Algorithmic change introduced a bug, or edge case not handled.",
                "suggested_action": "Use run_function_with_inputs to find exact inputs where behavior differs.",
                "priority": 1,
            })
        elif "memory" in veto_lower:
            hypotheses.append({
                "hypothesis": "The optimization uses more memory than the original.",
                "likely_cause": "New data structures (dict, set) have higher memory overhead than original approach.",
                "suggested_action": "Try an in-place approach or use generators instead of materializing collections.",
                "priority": 2,
            })
        elif "runtime" in veto_lower:
            hypotheses.append({
                "hypothesis": "The optimization is slower than the original.",
                "likely_cause": "Added overhead (function calls, data structure creation) exceeds savings, or optimization only helps at larger input sizes.",
                "suggested_action": "Use benchmark_function at multiple sizes to understand the crossover point.",
                "priority": 2,
            })
        elif "targeted" in veto_lower:
            hypotheses.append({
                "hypothesis": "Edge cases exposed by targeted tests are not handled correctly.",
                "likely_cause": "The optimization doesn't handle empty inputs, single elements, or boundary conditions.",
                "suggested_action": "Specifically test with empty input, single element, and boundary values.",
                "priority": 1,
            })
        elif "no measurable improvement" in veto_lower:
            hypotheses.append({
                "hypothesis": "The optimization doesn't measurably improve any metric.",
                "likely_cause": "The change is too minor, or the improvement is at a scale not captured by metrics.",
                "suggested_action": "Try a more aggressive optimization strategy — algorithmic change rather than micro-optimization.",
                "priority": 3,
            })

    if not hypotheses:
        hypotheses.append({
            "hypothesis": "Unknown failure mode — investigate with tools.",
            "likely_cause": "Need more data.",
            "suggested_action": "Use compare_metrics_detailed and analyze_code_diff to investigate.",
            "priority": 3,
        })

    hypotheses.sort(key=lambda h: h["priority"])
    return {
        "hypotheses": hypotheses,
        "total": len(hypotheses),
        "previous_hypothesis": hypothesis_from_previous_round,
    }


# ═══════════════════════════════════════════════════════════
# TOOL DISPATCHER — Routes tool calls to handlers
# ═══════════════════════════════════════════════════════════

TOOL_HANDLERS: dict[str, Callable] = {
    "run_lizard": handle_run_lizard,
    "run_radon": handle_run_radon,
    "analyze_bytecode": handle_analyze_bytecode,
    "benchmark_function": handle_benchmark_function,
    "estimate_big_o": handle_estimate_big_o,
    "run_function_with_inputs": handle_run_function_with_inputs,
    "compile_check": handle_compile_check,
    "compare_outputs": handle_compare_outputs,
    "run_tests": handle_run_tests,
    "quick_benchmark": handle_quick_benchmark,
    "web_research": handle_web_research,
    "lookup_optimization_pattern": handle_pattern_lookup,
    "analyze_code_diff": handle_diff_analysis,
    "compare_metrics_detailed": handle_metric_comparison,
    "generate_failure_hypotheses": handle_failure_hypotheses,
}


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """
    Execute a tool and return the result as a JSON string.
    This is called by the agentic loop when Claude requests a tool.

    Handles both sync and async handlers transparently.
    """
    handler = TOOL_HANDLERS.get(tool_name)
    if not handler:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    try:
        import asyncio
        import inspect

        if inspect.iscoroutinefunction(handler):
            # Handle async tool handlers
            try:
                loop = asyncio.get_running_loop()
                # If we're already in an event loop, create a task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(
                        asyncio.run, handler(**tool_input)
                    ).result()
            except RuntimeError:
                result = asyncio.run(handler(**tool_input))
        else:
            result = handler(**tool_input)

        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": f"Tool execution failed: {str(e)}"})


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════

def _categorize_opcode(opname: str) -> str:
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


def _top_opcodes(instructions, n=5) -> list:
    counts = {}
    for instr in instructions:
        counts[instr.opname] = counts.get(instr.opname, 0) + 1
    return sorted(counts.items(), key=lambda x: -x[1])[:n]


def _slope_to_big_o(slope: float) -> str:
    if slope < 0.1: return "O(1)"
    elif slope < 0.6: return "O(log n)"
    elif slope < 1.2: return "O(n)"
    elif slope < 1.6: return "O(n log n)"
    elif slope < 2.2: return "O(n²)"
    elif slope < 3.2: return "O(n³)"
    else: return f"O(n^{slope:.1f})"
