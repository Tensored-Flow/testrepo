"""Shared data contracts for the agent pipeline.

Every agent consumes and produces these types. This is the single source
of truth for what data flows between pipeline stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Triage ──────────────────────────────────────────────────

class Category(str, Enum):
    """Function triage classification."""
    A = "A"  # Pure, benchmarkable — no side effects
    B = "B"  # Side effects (DB, file I/O, network, GPU)
    C = "C"  # Skip (too trivial, generated, config, etc.)


@dataclass
class TargetFunction:
    """A single function identified by the triage agent."""
    file_path: str
    function_name: str
    source_code: str
    start_line: int
    end_line: int
    category: Category
    category_reason: str
    cyclomatic_complexity: int = 0
    has_tests: bool = False
    test_file: str | None = None


@dataclass
class TriageResult:
    """Output of the triage agent — all classified functions in a repo."""
    repo_path: str
    targets: list[TargetFunction] = field(default_factory=list)
    test_framework: str = "pytest"          # "pytest" | "unittest"
    total_functions_scanned: int = 0
    category_a_count: int = 0
    category_b_count: int = 0
    category_c_count: int = 0


# ── Behavioral Snapshot ─────────────────────────────────────

@dataclass
class BenchmarkPoint:
    """Single data point from benchmarking at one input size."""
    input_size: int
    time_ms: float
    memory_mb: float


@dataclass
class BehavioralSnapshot:
    """Complete behavioral fingerprint of a function before/after optimization."""
    function_name: str
    source_code: str
    static_metrics: dict[str, Any] = field(default_factory=dict)
    benchmarks: list[BenchmarkPoint] = field(default_factory=list)
    big_o: dict[str, Any] = field(default_factory=dict)
    input_output_pairs: list[tuple[Any, Any]] = field(default_factory=list)


# ── Analyst ─────────────────────────────────────────────────

@dataclass
class AnalysisHypothesis:
    """LLM-generated hypothesis about what's wrong and how to fix it."""
    function_name: str
    category: str                       # e.g. "algorithmic", "data_structure", "redundancy"
    summary: str                        # one-line description
    current_big_o: str                  # e.g. "O(n^2)"
    proposed_big_o: str                 # e.g. "O(n log n)"
    approach: str                       # detailed optimization strategy
    confidence: float = 0.0            # 0.0-1.0
    risks: list[str] = field(default_factory=list)


# ── Optimizer ───────────────────────────────────────────────

@dataclass
class OptimizedCode:
    """LLM-generated optimized version of a function."""
    function_name: str
    original_source: str
    optimized_source: str
    hypothesis: AnalysisHypothesis
    changes_summary: str                # human-readable diff description


# ── Test Designer ───────────────────────────────────────────

@dataclass
class TestSuite:
    """Generated tests for validating an optimization."""
    function_name: str
    differential_test_code: str         # compares original vs optimized outputs
    targeted_test_code: str             # edge cases specific to the optimization
    test_file_path: str | None = None


# ── Validator ───────────────────────────────────────────────

@dataclass
class TestResults:
    """Aggregated test execution results."""
    existing_tests_passed: bool = False
    differential_tests_passed: bool = False
    targeted_tests_passed: bool = False
    differential_failures: list[str] = field(default_factory=list)
    targeted_failures: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Final verdict from the validator agent."""
    function_name: str
    verdict: str                        # "APPROVED" | "REJECTED"
    vetoes: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    rejection_reason: str | None = None
    before_snapshot: BehavioralSnapshot | None = None
    after_snapshot: BehavioralSnapshot | None = None
    test_results: TestResults | None = None
