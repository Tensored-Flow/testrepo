"""
End-to-end integration test for the ComplexityImprover pipeline.
Tests all deterministic components without requiring an API key.
"""
import json
import sys
import traceback

# Ensure we're running from ai-agents/
sys.path.insert(0, ".")

passed = 0
failed = 0

def test(name, func):
    global passed, failed
    try:
        func()
        print(f"  ✅ {name}")
        passed += 1
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        traceback.print_exc()
        failed += 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST 1: TRIAGE — Full AST scan on demo repo
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n═══ TEST 1: TRIAGE ═══")
from backend.agents.triage import triage_agent

triage_result = triage_agent("demo_repo")

def test_triage_returns_dict():
    assert isinstance(triage_result, dict), f"Expected dict, got {type(triage_result)}"

def test_triage_has_targets():
    targets = triage_result.get("targets", [])
    assert len(targets) > 0, f"Expected optimization targets, got {len(targets)}"
    print(f"    → Found {len(targets)} Category A targets")
    for t in targets:
        print(f"      • {t['function_name']}() CC={t.get('cyclomatic_complexity', '?')}")

def test_triage_has_analysis_only():
    analysis = triage_result.get("analysis_only", [])
    print(f"    → Found {len(analysis)} Category B (analysis only)")
    for a in analysis:
        print(f"      • {a['function_name']}(): {a.get('reason', '?')}")

def test_triage_has_skipped():
    skipped = triage_result.get("skipped", [])
    print(f"    → Found {len(skipped)} Category C (skipped)")
    for s in skipped:
        print(f"      • {s.get('function_name', '?')}(): {s.get('reason', '?')}")

test("triage returns dict", test_triage_returns_dict)
test("triage finds targets", test_triage_has_targets)
test("triage finds analysis_only", test_triage_has_analysis_only)
test("triage finds skipped", test_triage_has_skipped)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST 2: SNAPSHOT — Full behavioral fingerprint
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n═══ TEST 2: SNAPSHOT ═══")
from backend.agents.snapshot import snapshot_agent, lightweight_snapshot, compute_static_metrics
from backend.models.types import TargetFunction, Category, BehavioralSnapshot
from backend.models.events import EventEmitter, EventType

# Pick a single-param target for full snapshot (bubble_sort, not matrix_multiply)
targets = triage_result.get("targets", [])
# Prefer bubble_sort since it takes a single list — benchmarks will work
t = None
for _t in targets:
    if _t["function_name"] == "bubble_sort":
        t = _t
        break
if t is None and targets:
    t = targets[0]
if t:
    target = TargetFunction(
        name=t["function_name"],
        file_path=t["file_path"],
        source_code=t["source_code"],
        start_line=t.get("start_line", 0),
        end_line=t.get("end_line", 0),
        cyclomatic_complexity=t.get("cyclomatic_complexity", 0),
        parameters=t.get("parameters", []),
        category=Category.A,
        reason=t.get("reason", ""),
    )

    emitter = EventEmitter()
    events_received = []
    emitter.on_event(lambda e: events_received.append(e))

    snapshot = snapshot_agent(target, emitter, skip_benchmarks=False)

    def test_snapshot_type():
        assert isinstance(snapshot, BehavioralSnapshot), f"Wrong type: {type(snapshot)}"

    def test_snapshot_metrics():
        sm = snapshot.static_metrics
        print(f"    → CC={sm.cyclomatic_complexity}, MI={sm.maintainability_index:.1f}, "
              f"Halstead D={sm.halstead_difficulty:.1f}, LOC={sm.loc}")
        assert sm.loc > 0, "LOC should be > 0"

    def test_snapshot_bytecode():
        assert snapshot.bytecode_instruction_count > 0, "No bytecode instructions"
        print(f"    → Bytecode: {snapshot.bytecode_instruction_count} instructions, "
              f"categories: {snapshot.bytecode_categories}")

    def test_snapshot_io_pairs():
        assert len(snapshot.input_output_pairs) > 0, "No I/O pairs captured"
        print(f"    → {len(snapshot.input_output_pairs)} I/O pairs captured")

    def test_snapshot_big_o():
        assert snapshot.big_o_estimate != "unknown", f"Big O unknown"
        print(f"    → Big O: {snapshot.big_o_estimate} (slope={snapshot.big_o_slope:.3f})")

    def test_snapshot_benchmarks():
        assert len(snapshot.benchmarks) > 0, "No benchmarks"
        for b in snapshot.benchmarks:
            print(f"    → n={b.input_size}: {b.mean_time*1000:.2f}ms, mem={b.memory_bytes}B")

    def test_snapshot_emits_events():
        assert len(events_received) > 0, "No events emitted"
        event_types = set(e.event_type.value for e in events_received)
        print(f"    → {len(events_received)} events emitted, types: {event_types}")

    test("snapshot returns BehavioralSnapshot", test_snapshot_type)
    test("snapshot has static metrics", test_snapshot_metrics)
    test("snapshot has bytecode analysis", test_snapshot_bytecode)
    test("snapshot has I/O pairs", test_snapshot_io_pairs)
    test("snapshot has Big O estimate", test_snapshot_big_o)
    test("snapshot has benchmarks", test_snapshot_benchmarks)
    test("snapshot emits events", test_snapshot_emits_events)
else:
    print("  ⚠️ No targets found — skipping snapshot tests")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST 3: LIGHTWEIGHT SNAPSHOT (Category B)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n═══ TEST 3: LIGHTWEIGHT SNAPSHOT (Category B) ═══")
analysis_only = triage_result.get("analysis_only", [])
if analysis_only:
    b = analysis_only[0]
    b_target = TargetFunction(
        name=b["function_name"],
        file_path=b["file_path"],
        source_code=b.get("source_code", ""),
        start_line=0, end_line=0,
        cyclomatic_complexity=b.get("cyclomatic_complexity", 0),
        parameters=b.get("parameters", []),
        category=Category.B,
        reason=b.get("reason", "Side effects"),
        red_flags=b.get("red_flags", []),
    )

    b_emitter = EventEmitter()
    lw_result = lightweight_snapshot(b_target, b_emitter)

    def test_lw_returns_dict():
        assert isinstance(lw_result, dict), f"Expected dict, got {type(lw_result)}"

    def test_lw_has_metrics():
        sm = lw_result.get("static_metrics", {})
        assert sm.get("loc", 0) > 0, "LOC should be > 0"
        print(f"    → {b_target.name}(): CC={sm.get('cyclomatic_complexity')}, "
              f"MI={sm.get('maintainability_index')}, LOC={sm.get('loc')}")

    def test_lw_has_bytecode():
        bc = lw_result.get("bytecode", {})
        print(f"    → Bytecode: {bc.get('instruction_count', 0)} instructions")

    test("lightweight_snapshot returns dict", test_lw_returns_dict)
    test("lightweight_snapshot has metrics", test_lw_has_metrics)
    test("lightweight_snapshot has bytecode", test_lw_has_bytecode)
else:
    print("  ⚠️ No Category B functions found — skipping")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST 4: TOOLS — All 10 tool handlers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n═══ TEST 4: TOOLS ═══")
from backend.core.tools import execute_tool, ALL_TOOLS, TOOL_HANDLERS

def test_run_lizard():
    r = json.loads(execute_tool("run_lizard", {"source_code": "def foo(x):\n    if x > 0:\n        return x\n    return -x"}))
    assert r.get("cyclomatic_complexity", 0) >= 1
    print(f"    → CC={r['cyclomatic_complexity']}")

def test_run_radon():
    r = json.loads(execute_tool("run_radon", {"source_code": "def foo(x):\n    if x > 0:\n        return x\n    return -x"}))
    assert "maintainability_index" in r
    print(f"    → MI={r['maintainability_index']}, Halstead V={r['halstead_volume']}")

def test_analyze_bytecode():
    code = "def foo(x):\n    return x * 2 + 1"
    r = json.loads(execute_tool("analyze_bytecode", {"source_code": code, "function_name": "foo"}))
    assert r.get("total_instructions", 0) > 0
    print(f"    → {r['total_instructions']} instructions, categories: {r.get('categories', {})}")

def test_compile_check_valid():
    r = json.loads(execute_tool("compile_check", {"source_code": "def bar(a, b): return a + b", "function_name": "bar"}))
    assert r["valid"] == True
    print(f"    → valid={r['valid']}, params={r.get('parameters')}")

def test_compile_check_invalid():
    r = json.loads(execute_tool("compile_check", {"source_code": "def bar(a, b: return a + b", "function_name": "bar"}))
    assert r["valid"] == False
    print(f"    → valid={r['valid']}, error={r.get('error', '')[:60]}")

def test_run_function_with_inputs():
    code = "def double(x):\n    return x * 2"
    inputs = [{"args": [1], "kwargs": {}}, {"args": [5], "kwargs": {}}, {"args": [0], "kwargs": {}}]
    r = json.loads(execute_tool("run_function_with_inputs", {
        "source_code": code, "function_name": "double", "test_inputs": inputs
    }))
    assert r["total_tests"] == 3
    print(f"    → {r['total_tests']} inputs tested, results: {[x.get('output') for x in r.get('results', [])]}")

def test_benchmark_function():
    code = "def sum_list(arr):\n    total = 0\n    for x in arr:\n        total += x\n    return total"
    gen = "import random\ndef generate_input(n):\n    return ([random.randint(0, 100) for _ in range(n)],)"
    r = json.loads(execute_tool("benchmark_function", {
        "source_code": code, "function_name": "sum_list",
        "input_generator": gen, "sizes": [50, 200]
    }))
    benchmarks = r.get("benchmarks", [])
    assert len(benchmarks) > 0, f"No benchmarks returned: {r}"
    valid = [b for b in benchmarks if isinstance(b, dict) and b.get("mean_time", -1) > 0]
    assert len(valid) > 0, f"No valid benchmarks: {benchmarks}"
    for b in valid:
        print(f"    → n={b['size']}: {b['mean_time']*1000:.3f}ms")

def test_estimate_big_o():
    r = json.loads(execute_tool("estimate_big_o", {
        "sizes": [100, 500, 1000, 5000],
        "times": [0.001, 0.005, 0.01, 0.05]
    }))
    assert r.get("big_o") != "unknown"
    print(f"    → {r['big_o']} (slope={r['slope']}, R²={r.get('r_squared')})")

def test_compare_outputs():
    orig = "def inc(x): return x + 1"
    opt = "def inc(x): return x + 1"  # same
    inputs = [{"args": [0], "kwargs": {}}, {"args": [5], "kwargs": {}}, {"args": [-1], "kwargs": {}}]
    r = json.loads(execute_tool("compare_outputs", {
        "original_source": orig, "optimized_source": opt,
        "function_name": "inc", "test_inputs": inputs
    }))
    assert r["all_match"] == True
    print(f"    → all_match={r['all_match']}, {r['matches']}/{r['total_tests']} matched")

def test_run_tests():
    test_code = """
def add(a, b):
    return a + b

def test_add_basic():
    assert add(1, 2) == 3

def test_add_zero():
    assert add(0, 0) == 0

if __name__ == "__main__":
    test_add_basic()
    test_add_zero()
    print("ALL TESTS PASSED")
"""
    r = json.loads(execute_tool("run_tests", {"test_code": test_code}))
    assert r.get("passed") == True or r.get("success") == True
    print(f"    → passed={r.get('passed')}, success={r.get('success')}")

test("run_lizard", test_run_lizard)
test("run_radon", test_run_radon)
test("analyze_bytecode", test_analyze_bytecode)
test("compile_check (valid)", test_compile_check_valid)
test("compile_check (invalid)", test_compile_check_invalid)
test("run_function_with_inputs", test_run_function_with_inputs)
test("benchmark_function", test_benchmark_function)
test("estimate_big_o", test_estimate_big_o)
test("compare_outputs", test_compare_outputs)
test("run_tests", test_run_tests)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST 5: VALIDATOR — Deterministic rubric
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n═══ TEST 5: VALIDATOR ═══")
from backend.agents.validator import compute_rubric, validator_agent
from backend.models.types import (
    BehavioralSnapshot, StaticMetrics, BenchmarkPoint,
    OptimizedCode, TestSuite, TestResults, Verdict,
)

# Simulate before/after for rubric test
before_metrics = StaticMetrics(
    cyclomatic_complexity=8, cognitive_complexity=12,
    maintainability_index=40.0, halstead_difficulty=15.0,
    halstead_bugs=0.5, loc=20, nesting_depth=3,
)
after_metrics = StaticMetrics(
    cyclomatic_complexity=3, cognitive_complexity=4,
    maintainability_index=65.0, halstead_difficulty=8.0,
    halstead_bugs=0.2, loc=12, nesting_depth=1,
)

before_snapshot = BehavioralSnapshot(
    function_name="test_func", file_path="test.py",
    source_code="def test_func(x): pass",
    static_metrics=before_metrics,
    benchmarks=[BenchmarkPoint(100, 0.01, 0.001, 1000), BenchmarkPoint(1000, 0.1, 0.01, 5000)],
    big_o_estimate="O(n)", big_o_slope=1.0,
)

# Tests pass, no vetoes
diff_results = TestResults(passed=5, failed=0)
targeted_results = TestResults(passed=3, failed=0)

def test_rubric_approved():
    verdict, vetoes, improvements = compute_rubric(
        before=before_snapshot,
        after_metrics=after_metrics,
        after_benchmarks=[BenchmarkPoint(100, 0.005, 0.001, 800), BenchmarkPoint(1000, 0.04, 0.005, 3000)],
        after_big_o_slope=0.9,
        differential_results=diff_results,
        targeted_results=targeted_results,
    )
    assert verdict == Verdict.APPROVED, f"Expected APPROVED, got {verdict}, vetoes: {vetoes}"
    improved = [m.metric_name for m in improvements if m.improved]
    print(f"    → {verdict.value}: improved {improved}")

def test_rubric_rejected_runtime():
    # After is SLOWER — should veto
    verdict, vetoes, improvements = compute_rubric(
        before=before_snapshot,
        after_metrics=after_metrics,
        after_benchmarks=[BenchmarkPoint(100, 0.02, 0.001, 800), BenchmarkPoint(1000, 0.15, 0.01, 5000)],
        after_big_o_slope=1.1,
        differential_results=diff_results,
        targeted_results=targeted_results,
    )
    assert verdict == Verdict.REJECTED, f"Expected REJECTED, got {verdict}"
    print(f"    → {verdict.value}: {vetoes}")

def test_rubric_rejected_test_fail():
    bad_tests = TestResults(passed=2, failed=3, errors=["AssertionError"])
    verdict, vetoes, improvements = compute_rubric(
        before=before_snapshot,
        after_metrics=after_metrics,
        after_benchmarks=[],
        after_big_o_slope=0.5,
        differential_results=bad_tests,
        targeted_results=TestResults(passed=1, failed=0),
    )
    assert verdict == Verdict.REJECTED
    assert any("Differential" in v for v in vetoes)
    print(f"    → {verdict.value}: {vetoes}")

test("rubric APPROVES good optimization", test_rubric_approved)
test("rubric REJECTS runtime regression", test_rubric_rejected_runtime)
test("rubric REJECTS test failures", test_rubric_rejected_test_fail)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST 6: REPORT GENERATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n═══ TEST 6: REPORT GENERATOR ═══")
from backend.agents.report_generator import report_generator_agent, format_report_as_pr_body
from backend.models.types import (
    RunResult, PipelineResult, ValidationResult, MetricsComparison,
    AnalysisHypothesis,
)

# Build a mock RunResult
mock_run = RunResult(run_id="test-123", repo_path="demo_repo")
mock_run.total_functions = 3
mock_run.optimized_count = 1
mock_run.rejected_count = 1
mock_run.error_count = 0

# One approved result
approved_validation = ValidationResult(
    function_name="bubble_sort",
    verdict=Verdict.APPROVED,
    improvements=[
        MetricsComparison("Cyclomatic Complexity", 8, 3, True, -62.5),
        MetricsComparison("Maintainability Index", 40.0, 65.0, True, 62.5),
        MetricsComparison("Runtime", 0.1, 0.04, True, -60.0),
    ],
    before_snapshot=before_snapshot,
    after_snapshot=BehavioralSnapshot(
        function_name="bubble_sort", file_path="test.py",
        source_code="def bubble_sort(arr): ...",
        static_metrics=after_metrics,
        big_o_estimate="O(n log n)", big_o_slope=1.1,
    ),
    test_results=TestResults(passed=8, failed=0),
)
mock_run.results.append(PipelineResult(
    function_name="bubble_sort",
    file_path="algorithms.py",
    validation=approved_validation,
    optimized_code=OptimizedCode(
        function_name="bubble_sort",
        original_source="def bubble_sort(arr): ...",
        optimized_source="def bubble_sort(arr): # optimized ...",
        changes_description="Replaced bubble sort with merge sort",
        strategy_used="divide-and-conquer",
    ),
    hypothesis=AnalysisHypothesis(
        function_name="bubble_sort",
        current_complexity="O(n²)",
        proposed_complexity="O(n log n)",
        bottleneck="Nested loop comparison",
        strategy="Replace with merge sort",
        expected_speedup="2-10x",
    ),
    rounds_taken=1,
    status="approved",
))

# One rejected result
mock_run.results.append(PipelineResult(
    function_name="find_duplicates",
    file_path="algorithms.py",
    validation=ValidationResult(
        function_name="find_duplicates",
        verdict=Verdict.REJECTED,
        veto_reasons=["VETO: Runtime regression +15.2%"],
    ),
    hypothesis=AnalysisHypothesis(
        function_name="find_duplicates",
        current_complexity="O(n²)",
        proposed_complexity="O(n)",
        bottleneck="Nested loop",
        strategy="Use set for O(1) lookup",
        expected_speedup="5-10x",
    ),
    rounds_taken=3,
    status="rejected",
))

# Add Category B reports
mock_run.category_b_reports = [
    {
        "function_name": "write_log",
        "file_path": "algorithms.py",
        "category": "B",
        "reason": "File I/O detected",
        "red_flags": ["open()", "write()"],
        "metrics": {"static_metrics": {"cyclomatic_complexity": 2, "loc": 5}},
        "recommendation": "Manual optimization recommended",
    }
]

# Generate report (without Haiku — will use fallback)
report_emitter = EventEmitter()
report = report_generator_agent(mock_run, report_emitter, include_executive_summary=False)

def test_report_has_summary():
    s = report.get("summary", {})
    assert s["total_targets"] == 3
    assert s["optimized"] == 1
    assert s["rejected"] == 1
    print(f"    → Summary: {s['optimized']} optimized, {s['rejected']} rejected, {s['total_rounds_used']} rounds")

def test_report_has_optimizations():
    opts = report.get("optimizations", [])
    assert len(opts) == 1
    o = opts[0]
    assert o["confidence_score"] > 0
    print(f"    → {o['function_name']}(): confidence={o['confidence_score']}%, strategy={o['strategy']}")
    improved = [i for i in o.get("improvements", []) if i.get("improved")]
    print(f"    → {len(improved)} improved metrics")

def test_report_has_rejected():
    rej = report.get("rejected", [])
    assert len(rej) == 1
    print(f"    → Rejected: {rej[0]['function_name']}(), reasons: {rej[0].get('veto_reasons')}")

def test_report_has_category_b():
    b_reports = report.get("category_b_reports", [])
    assert len(b_reports) == 1
    print(f"    → Category B: {b_reports[0]['function_name']}(): {b_reports[0].get('reason')}")

def test_pr_body_format():
    pr_body = format_report_as_pr_body(report)
    assert "ComplexityImprover" in pr_body
    assert "bubble_sort" in pr_body
    assert "find_duplicates" in pr_body
    lines = pr_body.split("\n")
    print(f"    → PR body: {len(lines)} lines")
    # Print first few lines
    for line in lines[:8]:
        print(f"    | {line}")

test("report has summary stats", test_report_has_summary)
test("report has optimizations", test_report_has_optimizations)
test("report has rejected functions", test_report_has_rejected)
test("report has Category B reports", test_report_has_category_b)
test("PR body format is valid", test_pr_body_format)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST 7: EVENT EMITTER (all event types)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n═══ TEST 7: EVENT EMITTER ═══")

def test_all_event_types():
    em = EventEmitter()
    events = []
    em.on_event(lambda e: events.append(e))

    em.log("test", "hello")
    em.complete(EventType.ANALYSIS_COMPLETE, "analyst", "done")
    em.error("test", "oops")
    em.tool_call("analyst", "run_lizard", {"source_code": "..."})
    em.tool_result("analyst", "run_lizard", "CC=5")
    em.thinking("analyst", "Let me think about this...")

    assert len(events) == 6
    types = [e.event_type for e in events]
    assert EventType.AGENT_LOG in types
    assert EventType.ANALYSIS_COMPLETE in types
    assert EventType.AGENT_ERROR in types
    assert EventType.TOOL_CALL in types
    assert EventType.TOOL_RESULT in types
    assert EventType.THINKING in types

    # Test SSE serialization
    for e in events:
        sse = e.to_sse()
        assert sse.startswith("data: ")
        assert sse.endswith("\n\n")
        payload = json.loads(sse[6:].strip())
        assert "event_type" in payload
        assert "agent" in payload
    print(f"    → {len(events)} events, all serialize to valid SSE")

test("all event types work + SSE serialization", test_all_event_types)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEST 8: SERVER — Endpoints respond correctly
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n═══ TEST 8: SERVER ENDPOINTS ═══")
from fastapi.testclient import TestClient
from backend.server import app

client = TestClient(app)

def test_health_endpoint():
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "tool_use" in data["features"]
    assert "extended_thinking" in data["features"]
    assert "self_validation" in data["features"]
    assert "strategic_planning" in data["features"]
    assert "sse_streaming" in data["features"]
    print(f"    → status={data['status']}, features={data['features']}")

def test_results_404():
    r = client.get("/api/results/nonexistent")
    assert r.status_code == 404

def test_optimize_endpoint_exists():
    # Don't actually start a run (needs API key), just verify endpoint exists
    r = client.post("/api/optimize", json={"repo_path": "/fake"})
    # Should get 200 (starts background task) even without API key
    # The actual pipeline will fail internally but the endpoint responds
    assert r.status_code == 200
    data = r.json()
    assert "run_id" in data
    print(f"    → run_id={data['run_id'][:8]}..., message={data['message']}")

test("GET /api/health", test_health_endpoint)
test("GET /api/results/nonexistent → 404", test_results_404)
test("POST /api/optimize responds", test_optimize_endpoint_exists)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FINAL SCORE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print(f"\n{'━' * 50}")
print(f"  RESULTS: {passed} passed, {failed} failed")
print(f"{'━' * 50}")

if failed > 0:
    sys.exit(1)
else:
    print("  🎉 ALL INTEGRATION TESTS PASSED")
