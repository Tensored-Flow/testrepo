"""Agent 2: Snapshot — behavioral fingerprinting (deterministic).

Captures a complete behavioral profile of a function BEFORE optimization.
This becomes the ground truth that the validator checks against.

No LLM calls. Pure computation.

Owner: ___
Status: NOT STARTED
Depends on: core/metrics_engine.py, core/behavioral_snapshot.py, core/sandbox.py, models/types.py
"""

from __future__ import annotations

from backend.models.types import TargetFunction, BehavioralSnapshot


def snapshot_agent(target: TargetFunction) -> BehavioralSnapshot:
    """Capture a full behavioral snapshot of a function.

    Pipeline position: SECOND (runs after triage, before analyst)
    Input: a single TargetFunction (Category A only)
    Output: BehavioralSnapshot with metrics, benchmarks, I/O pairs

    Owner: ___
    Status: NOT STARTED
    Depends on: core/metrics_engine.py, core/behavioral_snapshot.py, core/sandbox.py
    """
    # TODO: Run compute_static_metrics on target.source_code
    # TODO: Compile the function via sandbox.compile_function
    # TODO: Generate deterministic test inputs via behavioral_snapshot.generate_test_inputs
    # TODO: Run benchmark_function at standard sizes [100, 500, 1000, 5000]
    # TODO: Run estimate_big_o on benchmark results
    # TODO: Capture input→output pairs for differential testing:
    #   - Run function on each test input, record (input, output)
    #   - Use copy.deepcopy to preserve inputs
    # TODO: Return populated BehavioralSnapshot
    pass
