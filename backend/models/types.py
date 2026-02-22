"""
Shared data contracts for the ComplexityImprover pipeline.
All agents communicate through these dataclasses.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Category(str, Enum):
    A = "A"  # Pure, benchmarkable → optimize
    B = "B"  # Side effects → analysis only
    C = "C"  # Skip (tests, boilerplate, trivial)


class Verdict(str, Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


# ─────────────────────────────────────────────
# Triage
# ─────────────────────────────────────────────

@dataclass
class TargetFunction:
    name: str
    file_path: str              # relative to repo root
    source_code: str
    start_line: int
    end_line: int
    cyclomatic_complexity: int
    parameters: list[str]
    category: Category
    reason: str
    red_flags: list[str] = field(default_factory=list)


@dataclass
class TriageResult:
    category_a: list[TargetFunction]
    category_b: list[TargetFunction]
    category_c: list[TargetFunction]
    repo_path: str
    test_framework: Optional[str] = None   # "pytest" | "unittest" | None
    test_dir: Optional[str] = None


# ─────────────────────────────────────────────
# Snapshot (Agent 2)
# ─────────────────────────────────────────────

@dataclass
class BenchmarkPoint:
    input_size: int
    mean_time: float        # seconds
    std_time: float         # seconds
    memory_bytes: int = 0


@dataclass
class StaticMetrics:
    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0
    halstead_volume: float = 0.0
    halstead_difficulty: float = 0.0
    halstead_effort: float = 0.0
    halstead_bugs: float = 0.0
    maintainability_index: float = 0.0
    loc: int = 0
    nesting_depth: int = 0
    parameter_count: int = 0


@dataclass
class BehavioralSnapshot:
    function_name: str
    file_path: str
    source_code: str
    static_metrics: StaticMetrics
    benchmarks: list[BenchmarkPoint] = field(default_factory=list)
    big_o_estimate: str = "unknown"         # e.g. "O(n^2)", "O(n log n)"
    big_o_slope: float = 0.0                # log-log regression slope
    input_output_pairs: list[dict] = field(default_factory=list)
    # Each pair: {"input": ..., "output": ..., "output_type": str}
    bytecode_instruction_count: int = 0
    bytecode_categories: dict = field(default_factory=dict)


# ─────────────────────────────────────────────
# Analyst (Agent 3)
# ─────────────────────────────────────────────

@dataclass
class AnalysisHypothesis:
    function_name: str
    current_complexity: str         # e.g. "O(n^2)"
    proposed_complexity: str        # e.g. "O(n log n)"
    bottleneck: str                 # human-readable diagnosis
    strategy: str                   # e.g. "Replace nested loops with hash map"
    expected_speedup: str           # e.g. "2-5x for large inputs"
    risks: list[str] = field(default_factory=list)
    round_number: int = 1
    failure_diagnosis: Optional[str] = None   # populated on round 2+


# ─────────────────────────────────────────────
# Optimizer (Agent 4)
# ─────────────────────────────────────────────

@dataclass
class OptimizedCode:
    function_name: str
    original_source: str
    optimized_source: str
    changes_description: str
    strategy_used: str
    round_number: int = 1


# ─────────────────────────────────────────────
# Test Designer (Agent 5)
# ─────────────────────────────────────────────

@dataclass
class TestSuite:
    function_name: str
    differential_tests: str         # Python test code comparing original vs optimized
    targeted_tests: str             # Python test code for edge cases
    test_count: int = 0


# ─────────────────────────────────────────────
# Validator (Agent 6)
# ─────────────────────────────────────────────

@dataclass
class TestResults:
    passed: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class MetricsComparison:
    metric_name: str
    before: float
    after: float
    improved: bool
    delta_percent: float


@dataclass
class ValidationResult:
    function_name: str
    verdict: Verdict
    improvements: list[MetricsComparison] = field(default_factory=list)
    veto_reasons: list[str] = field(default_factory=list)
    before_snapshot: Optional[BehavioralSnapshot] = None
    after_snapshot: Optional[BehavioralSnapshot] = None
    test_results: Optional[TestResults] = None
    round_number: int = 1


# ─────────────────────────────────────────────
# Pipeline-level
# ─────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Final output for one function through the full pipeline."""
    function_name: str
    file_path: str
    validation: Optional[ValidationResult] = None
    optimized_code: Optional[OptimizedCode] = None
    hypothesis: Optional[AnalysisHypothesis] = None
    rounds_taken: int = 0
    status: str = "pending"  # pending | approved | rejected | error


@dataclass
class RunResult:
    """Full pipeline run across all functions in a repo."""
    run_id: str
    repo_path: str
    results: list[PipelineResult] = field(default_factory=list)
    triage: Optional[TriageResult] = None
    category_b_reports: list[dict] = field(default_factory=list)
    total_functions: int = 0
    optimized_count: int = 0
    rejected_count: int = 0
    error_count: int = 0