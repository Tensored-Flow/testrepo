"""SSE event types for streaming agent progress to the frontend.

The frontend subscribes to an SSE endpoint and receives these events
as each agent completes its stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """All event types the frontend can receive."""
    PIPELINE_START = "pipeline_start"
    TRIAGE_COMPLETE = "triage_complete"
    SNAPSHOT_COMPLETE = "snapshot_complete"
    ANALYSIS_COMPLETE = "analysis_complete"
    OPTIMIZATION_COMPLETE = "optimization_complete"
    TESTS_GENERATED = "tests_generated"
    VALIDATION_COMPLETE = "validation_complete"
    PIPELINE_COMPLETE = "pipeline_complete"
    AGENT_ERROR = "agent_error"
    AGENT_LOG = "agent_log"


@dataclass
class AgentEvent:
    """A single event emitted by the pipeline for SSE streaming."""
    event_type: EventType
    agent_name: str
    data: dict[str, Any] = field(default_factory=dict)
    message: str = ""
