"""
Agent 6: Validator — Deterministic gatekeeper.

NO LLM CALLS. This is pure math.

Runs the full validation rubric:
  5 Hard Vetoes (any one = REJECTED):
    1. Existing test suite fails
    2. Behavioral mismatch (differential tests)
    3. Targeted tests fail
    4. Memory regression >20%
    5. Runtime regression >10%

  8 Improvement Checks (need >= 1 for APPROVED):
    CC decreased, cognitive complexity decreased, MI increased,
    Halstead difficulty decreased, estimated bugs decreased,
    runtime decreased >10%, memory decreased >10%, Big O slope decreased

"The AI proposes, the math disposes."
"""

from typing import Optional
from dataclasses import asdict

from backend.models.types import (
    BehavioralSnapshot, OptimizedCode, TestSuite,
    ValidationResult, TestResults, MetricsComparison, Verdict,
)
from backend.models.events import EventEmitter, EventType
from backend.agents.snapshot import snapshot_agent, compute_static_metrics, estimate_big_o
from backend.core.sandbox import run_tests, run_function_with_inputs, benchmark_function
from backend.core.conversation import ConversationContext

try:
    from scipy.stats import ttest_ind
except ImportError:
    ttest_ind = None


# ─────────────────────────────────────────────
# Test execution
# ─────────────────────────────────────────────

def _run_test_suite(
    test_code: str,
    timeout: float = 15.0,
) -> TestResults:
    """Run a test suite and count pass/fail."""
    if not test_code or not test_code.strip():
        return TestResults(passed=0, failed=0, errors=["No test code provided"])

    result = run_tests(test_code, timeout=timeout)

    if result.success:
        # Count test functions that ran
        test_count = test_code.count("def test_")
        return TestResults(passed=max(test_count, 1), failed=0, errors=[])
    else:
        # Parse errors from output
        errors = []
        if result.stderr:
            errors = [line for line in result.stderr.split("\n") if line.strip()][:5]
        if result.error:
            errors.append(result.error)
        return TestResults(passed=0, failed=1, errors=errors)


def _run_behavioral_comparison(
    original_source: str,
    optimized_source: str,
    function_name: str,
    original_io_pairs: list[dict],
) -> tuple[bool, list[str]]:
    """
    Re-run the original's input->output pairs through the optimized function.
    Return (all_match, mismatch_descriptions).
    """
    if not original_io_pairs:
        return True, []

    # Extract inputs that had no errors
    test_inputs = []
    expected_outputs = []
    for pair in original_io_pairs:
        if pair.get("error") is None and pair.get("input"):
            test_inputs.append(pair["input"])
            expected_outputs.append(pair.get("output"))

    if not test_inputs:
        return True, []

    # Run optimized function with same inputs
    optimized_results = run_function_with_inputs(
        optimized_source, function_name, test_inputs, timeout=10.0
    )

    mismatches = []
    for i, (expected, actual) in enumerate(zip(expected_outputs, optimized_results)):
        if isinstance(actual, dict) and actual.get("error"):
            mismatches.append(f"Input {i}: optimized raised {actual['error']}")
        elif isinstance(actual, dict) and actual.get("output") != expected:
            mismatches.append(
                f"Input {i}: expected {expected}, got {actual.get('output')}"
            )

    return len(mismatches) == 0, mismatches


# ─────────────────────────────────────────────
# Metrics comparison
# ─────────────────────────────────────────────

def _compare_metric(name: str, before: float, after: float, higher_is_better: bool = False) -> MetricsComparison:
    """Compare a before/after metric. Returns MetricsComparison."""
    if before == 0:
        delta_pct = 0.0
    else:
        delta_pct = ((after - before) / abs(before)) * 100

    if higher_is_better:
        improved = after > before
    else:
        improved = after < before

    return MetricsComparison(
        metric_name=name,
        before=before,
        after=after,
        improved=improved,
        delta_percent=delta_pct,
    )


# ─────────────────────────────────────────────
# Welch's t-test for benchmark significance
# ─────────────────────────────────────────────

def _run_significance_test(
    before_benchmarks: list,
    after_benchmarks: list,
) -> tuple[Optional[float], Optional[bool], list[dict]]:
    """
    Run per-size Welch's t-test on raw timing arrays.

    Returns:
        (worst_p_value, all_significant, per_size_results)
        where per_size_results is [{size, p_value, significant, before_n, after_n}, ...]
    """
    if ttest_ind is None:
        return None, None, []

    per_size = []
    for bb, ab in zip(before_benchmarks, after_benchmarks):
        before_times = bb.raw_times if hasattr(bb, 'raw_times') else []
        after_times = ab.raw_times if hasattr(ab, 'raw_times') else []

        if len(before_times) < 3 or len(after_times) < 3:
            per_size.append({
                "size": bb.input_size,
                "p_value": None,
                "significant": None,
                "before_n": len(before_times),
                "after_n": len(after_times),
            })
            continue

        t_stat, p_value = ttest_ind(before_times, after_times, equal_var=False)
        p_val = float(p_value)
        per_size.append({
            "size": bb.input_size,
            "p_value": round(p_val, 6),
            "significant": p_val < 0.05,
            "before_n": len(before_times),
            "after_n": len(after_times),
        })

    valid = [r for r in per_size if r["p_value"] is not None]
    if not valid:
        return None, None, per_size

    worst_p = max(r["p_value"] for r in valid)
    all_sig = all(r["significant"] for r in valid)
    return worst_p, all_sig, per_size


# ─────────────────────────────────────────────
# The Deterministic Rubric
# ─────────────────────────────────────────────

def compute_rubric(
    before: BehavioralSnapshot,
    after_metrics,
    after_benchmarks: list,
    after_big_o_slope: float,
    differential_results: TestResults,
    targeted_results: TestResults,
    existing_test_results: Optional[TestResults] = None,
) -> tuple[Verdict, list[str], list[MetricsComparison]]:
    """
    Pure function. 5 hard vetoes + 8 improvement checks.
    NO AI judgment. Returns (verdict, veto_reasons, improvements).
    """
    vetoes = []
    improvements = []

    bm = before.static_metrics
    am = after_metrics

    # ═══════════════════════════════════════════
    # 5 HARD VETOES
    # ═══════════════════════════════════════════

    # Veto 1: Existing test suite fails
    if existing_test_results and existing_test_results.failed > 0:
        vetoes.append(f"VETO: Existing test suite failed ({existing_test_results.failed} failures)")

    # Veto 2: Behavioral mismatch (differential tests)
    if differential_results.failed > 0:
        vetoes.append(f"VETO: Differential tests failed ({differential_results.failed} failures)")

    # Veto 3: Targeted tests fail
    if targeted_results.failed > 0:
        vetoes.append(f"VETO: Targeted tests failed ({targeted_results.failed} failures)")

    # Veto 4: Memory regression >20%
    if before.benchmarks and after_benchmarks:
        before_mem = max((b.memory_bytes for b in before.benchmarks), default=0)
        after_mem = max((b.memory_bytes for b in after_benchmarks), default=0)
        if before_mem > 0:
            mem_change = ((after_mem - before_mem) / before_mem) * 100
            if mem_change > 20:
                vetoes.append(f"VETO: Memory regression {mem_change:+.1f}% (limit: +20%)")

    # Veto 5: Runtime regression >10%
    if before.benchmarks and after_benchmarks:
        before_time = max((b.mean_time for b in before.benchmarks), default=0)
        after_time = max((b.mean_time for b in after_benchmarks), default=0)
        if before_time > 0:
            time_change = ((after_time - before_time) / before_time) * 100
            if time_change > 10:
                vetoes.append(f"VETO: Runtime regression {time_change:+.1f}% (limit: +10%)")

    # ═══════════════════════════════════════════
    # 8 IMPROVEMENT CHECKS (need >= 1)
    # ═══════════════════════════════════════════

    # 1. CC decreased
    cc = _compare_metric("Cyclomatic Complexity", bm.cyclomatic_complexity, am.cyclomatic_complexity)
    improvements.append(cc)

    # 2. Cognitive complexity decreased
    cog = _compare_metric("Cognitive Complexity", bm.cognitive_complexity, am.cognitive_complexity)
    improvements.append(cog)

    # 3. MI increased (higher is better)
    mi = _compare_metric("Maintainability Index", bm.maintainability_index, am.maintainability_index, higher_is_better=True)
    improvements.append(mi)

    # 4. Halstead difficulty decreased
    hd = _compare_metric("Halstead Difficulty", bm.halstead_difficulty, am.halstead_difficulty)
    improvements.append(hd)

    # 5. Estimated bugs decreased
    hb = _compare_metric("Estimated Bugs", bm.halstead_bugs, am.halstead_bugs)
    improvements.append(hb)

    # 6. Runtime decreased >10%
    if before.benchmarks and after_benchmarks:
        before_time = max((b.mean_time for b in before.benchmarks), default=0)
        after_time = max((b.mean_time for b in after_benchmarks), default=0)
        rt = _compare_metric("Runtime", before_time, after_time)
        # Only count as improvement if >10% decrease
        if rt.delta_percent < -10:
            rt.improved = True
        else:
            rt.improved = False
        improvements.append(rt)

    # 7. Memory decreased >10%
    if before.benchmarks and after_benchmarks:
        before_mem = max((b.memory_bytes for b in before.benchmarks), default=0)
        after_mem = max((b.memory_bytes for b in after_benchmarks), default=0)
        mem = _compare_metric("Memory", before_mem, after_mem)
        if mem.delta_percent < -10:
            mem.improved = True
        else:
            mem.improved = False
        improvements.append(mem)

    # 8. Big O slope decreased
    bo = _compare_metric("Big O Slope", before.big_o_slope, after_big_o_slope)
    improvements.append(bo)

    # ═══════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════

    if vetoes:
        return Verdict.REJECTED, vetoes, improvements

    has_improvement = any(m.improved for m in improvements)
    if not has_improvement:
        vetoes.append("No measurable improvement on any metric")
        return Verdict.REJECTED, vetoes, improvements

    return Verdict.APPROVED, [], improvements


# ─────────────────────────────────────────────
# Main Agent Entry Point
# ─────────────────────────────────────────────

def validator_agent(
    before_snapshot: BehavioralSnapshot,
    optimized: OptimizedCode,
    test_suite: TestSuite,
    emitter: Optional[EventEmitter] = None,
    skip_benchmarks: bool = False,
    existing_test_command: Optional[str] = None,
    ctx: Optional[ConversationContext] = None,
) -> ValidationResult:
    """
    Run the full deterministic validation rubric.

    Args:
        before_snapshot: The "before" behavioral snapshot
        optimized: The optimizer's output
        test_suite: The test designer's output
        emitter: Event emitter for SSE streaming
        skip_benchmarks: Skip timing (faster for demo)
        existing_test_command: Command to run existing tests
        ctx: Optional ConversationContext for multi-agent communication

    Returns:
        ValidationResult with verdict, improvements, and vetoes
    """
    fname = before_snapshot.function_name
    round_num = optimized.round_number
    _log = lambda msg: emitter.log("validator", msg, function_name=fname, round_number=round_num) if emitter else None

    _log(f"Validating optimization of {fname}() -- Round {round_num}")

    # -- Step 1: Run differential tests --
    _log(f"Running differential tests...")
    diff_results = _run_test_suite(test_suite.differential_tests)
    _log(f"Differential: {diff_results.passed} passed, {diff_results.failed} failed")

    # -- Step 2: Run targeted tests --
    _log(f"Running targeted tests...")
    targeted_results = _run_test_suite(test_suite.targeted_tests)
    _log(f"Targeted: {targeted_results.passed} passed, {targeted_results.failed} failed")

    # -- Step 3: Behavioral comparison (snapshot I/O pairs) --
    _log(f"Running behavioral comparison...")
    behavior_match, mismatches = _run_behavioral_comparison(
        before_snapshot.source_code,
        optimized.optimized_source,
        fname,
        before_snapshot.input_output_pairs,
    )
    if not behavior_match:
        _log(f"Behavioral mismatch detected: {len(mismatches)} differences")
        diff_results.failed += len(mismatches)
        diff_results.errors.extend(mismatches[:3])

    # -- Step 4: Compute "after" static metrics --
    _log(f"Computing after-metrics...")
    after_static = compute_static_metrics(optimized.optimized_source, before_snapshot.file_path)

    # -- Step 5: Benchmark optimized code (optional) --
    from backend.agents.snapshot import generate_input_generator_code, BenchmarkPoint
    after_benchmarks = []
    after_big_o_slope = 0.0

    if not skip_benchmarks and before_snapshot.benchmarks:
        _log(f"Benchmarking optimized {fname}()...")
        gen_code = generate_input_generator_code(fname, [])
        sizes = [b.input_size for b in before_snapshot.benchmarks]
        raw = benchmark_function(
            optimized.optimized_source, fname, gen_code,
            sizes=sizes, runs_per_size=10, timeout=30.0,
        )
        for b in raw:
            if isinstance(b, dict) and b.get("mean_time", -1) > 0:
                after_benchmarks.append(BenchmarkPoint(
                    input_size=b["size"],
                    mean_time=b["mean_time"],
                    std_time=b.get("std_time", 0.0),
                    memory_bytes=b.get("memory_bytes", 0),
                    raw_times=b.get("raw_times", []),
                ))
        if after_benchmarks:
            _, after_big_o_slope = estimate_big_o(after_benchmarks)

    # -- Step 5b: Welch's t-test on raw timing data --
    p_value = None
    significant = None
    p_values_per_size = []

    if before_snapshot.benchmarks and after_benchmarks:
        _log(f"Running Welch's t-test on timing data...")
        p_value, significant, p_values_per_size = _run_significance_test(
            before_snapshot.benchmarks, after_benchmarks,
        )
        if p_value is not None:
            _log(f"T-test: p={p_value:.6f}, significant={significant}")
        else:
            _log(f"T-test: insufficient raw timing data")

    # -- Step 6: Run existing test suite (if available) --
    existing_results = None

    # -- Step 7: RUBRIC --
    _log(f"Running deterministic rubric...")
    verdict, vetoes, improvements = compute_rubric(
        before=before_snapshot,
        after_metrics=after_static,
        after_benchmarks=after_benchmarks,
        after_big_o_slope=after_big_o_slope,
        differential_results=diff_results,
        targeted_results=targeted_results,
        existing_test_results=existing_results,
    )

    # Build after snapshot
    after_snapshot = BehavioralSnapshot(
        function_name=fname,
        file_path=before_snapshot.file_path,
        source_code=optimized.optimized_source,
        static_metrics=after_static,
        benchmarks=after_benchmarks,
        big_o_estimate=_slope_to_big_o_str(after_big_o_slope),
        big_o_slope=after_big_o_slope,
        input_output_pairs=[],
        bytecode_instruction_count=0,
        bytecode_categories={},
    )

    # Merge test results
    all_tests = TestResults(
        passed=diff_results.passed + targeted_results.passed,
        failed=diff_results.failed + targeted_results.failed,
        errors=diff_results.errors + targeted_results.errors,
    )

    result = ValidationResult(
        function_name=fname,
        verdict=verdict,
        improvements=improvements,
        veto_reasons=vetoes,
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
        test_results=all_tests,
        round_number=round_num,
        p_value=p_value,
        significant=significant,
        p_values_per_size=p_values_per_size,
    )

    # -- Write to conversation context if available --
    if ctx:
        ctx.validation_result = {
            "verdict": verdict.value,
            "vetoes": vetoes,
            "improvements": [
                {"metric": m.metric_name, "before": m.before, "after": m.after,
                 "improved": m.improved, "delta_percent": m.delta_percent}
                for m in improvements
            ],
            "p_value": p_value,
            "significant": significant,
            "p_values_per_size": p_values_per_size,
        }

        if verdict == Verdict.APPROVED:
            ctx.add_message(
                sender="validator",
                recipient="all",
                message_type="approval",
                content={
                    "final_metrics": ctx.validation_result,
                    "improvements": [m.metric_name for m in improvements if m.improved],
                    "tests_passed": all_tests.passed,
                },
            )
        else:
            ctx.add_message(
                sender="validator",
                recipient="analyst",
                message_type="rejection",
                content={
                    "vetoes": vetoes,
                    "metrics": ctx.validation_result,
                    "suggestion": "Form a different hypothesis that avoids these vetoes.",
                    "metric_details": {
                        m.metric_name: {"before": m.before, "after": m.after, "delta": m.delta_percent}
                        for m in improvements
                    },
                },
            )

    # -- Emit result --
    improved_metrics = [m.metric_name for m in improvements if m.improved]
    if verdict == Verdict.APPROVED:
        _log(f"APPROVED -- improved: {', '.join(improved_metrics) or 'metrics'}")
    else:
        _log(f"REJECTED -- {'; '.join(vetoes)}")

    if emitter:
        emitter.complete(
            EventType.VALIDATION_COMPLETE, "validator",
            f"{verdict.value}: {fname}()",
            function_name=fname,
            round_number=round_num,
            data={
                "verdict": verdict.value,
                "vetoes": vetoes,
                "improved_metrics": improved_metrics,
                "tests_passed": all_tests.passed,
                "tests_failed": all_tests.failed,
                "p_value": p_value,
                "significant": significant,
                "p_values_per_size": p_values_per_size,
            }
        )

    return result


def _slope_to_big_o_str(slope: float) -> str:
    if slope < 0.1: return "O(1)"
    elif slope < 0.6: return "O(log n)"
    elif slope < 1.2: return "O(n)"
    elif slope < 1.6: return "O(n log n)"
    elif slope < 2.2: return "O(n^2)"
    elif slope < 3.2: return "O(n^3)"
    else: return f"O(n^{slope:.1f})"
