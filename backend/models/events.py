"""
SSE Event Types — Updated for agentic tool use + extended thinking.

New event types:
  - TOOL_CALL: Agent is calling a deterministic tool
  - TOOL_RESULT: Tool returned a result
  - THINKING: Extended thinking trace from Claude
  - PLAN_COMPLETE: Planner agent finished strategic planning
  - SELF_VALIDATION: Optimizer/TestDesigner self-validated its output
"""

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EventType(str, Enum):
    # Pipeline lifecycle
    PIPELINE_START = "PIPELINE_START"
    PIPELINE_COMPLETE = "PIPELINE_COMPLETE"

    # Planner
    PLAN_COMPLETE = "PLAN_COMPLETE"

    # Per-agent completion
    TRIAGE_COMPLETE = "TRIAGE_COMPLETE"
    SNAPSHOT_COMPLETE = "SNAPSHOT_COMPLETE"
    ANALYSIS_COMPLETE = "ANALYSIS_COMPLETE"
    OPTIMIZATION_COMPLETE = "OPTIMIZATION_COMPLETE"
    TESTS_GENERATED = "TESTS_GENERATED"
    VALIDATION_COMPLETE = "VALIDATION_COMPLETE"

    # Agentic events (NEW — what judges want to see)
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"
    THINKING = "THINKING"
    SELF_VALIDATION = "SELF_VALIDATION"

    # Feedback loop
    ROUND_START = "ROUND_START"
    REJECTION_DIAGNOSIS = "REJECTION_DIAGNOSIS"

    # Inter-agent communication
    AGENT_HANDOFF = "AGENT_HANDOFF"

    # General
    AGENT_LOG = "AGENT_LOG"
    AGENT_ERROR = "AGENT_ERROR"


@dataclass
class PipelineEvent:
    event_type: EventType
    agent: str
    message: str
    data: Optional[dict] = None
    function_name: Optional[str] = None
    round_number: int = 1
    timestamp: float = field(default_factory=time.time)

    def to_sse(self) -> str:
        """Format as Server-Sent Event string."""
        payload = {
            "event_type": self.event_type.value,
            "agent": self.agent,
            "message": self.message,
            "function_name": self.function_name,
            "round_number": self.round_number,
            "timestamp": self.timestamp,
        }
        if self.data:
            payload["data"] = self.data
        return f"data: {json.dumps(payload)}\n\n"

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type.value,
            "agent": self.agent,
            "message": self.message,
            "function_name": self.function_name,
            "round_number": self.round_number,
            "timestamp": self.timestamp,
            "data": self.data,
        }


class EventEmitter:
    """
    Collects events during a pipeline run.
    Agents call emitter.log() / emitter.complete() / emitter.error() / emitter.tool_call().
    The server reads .events and streams them as SSE.
    """

    def __init__(self):
        self.events: list[PipelineEvent] = []
        self._callbacks: list = []

    def on_event(self, callback):
        """Register a callback for real-time streaming (e.g., SSE queue.put)."""
        self._callbacks.append(callback)

    def _emit(self, event: PipelineEvent):
        self.events.append(event)
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception:
                pass

    def log(self, agent: str, message: str, function_name: str = None,
            round_number: int = 1, data: dict = None):
        self._emit(PipelineEvent(
            event_type=EventType.AGENT_LOG,
            agent=agent,
            message=message,
            function_name=function_name,
            round_number=round_number,
            data=data,
        ))

    def complete(self, event_type: EventType, agent: str, message: str,
                 function_name: str = None, round_number: int = 1, data: dict = None):
        self._emit(PipelineEvent(
            event_type=event_type,
            agent=agent,
            message=message,
            function_name=function_name,
            round_number=round_number,
            data=data,
        ))

    def error(self, agent: str, message: str, function_name: str = None,
              round_number: int = 1):
        self._emit(PipelineEvent(
            event_type=EventType.AGENT_ERROR,
            agent=agent,
            message=message,
            function_name=function_name,
            round_number=round_number,
        ))

    def tool_call(self, agent: str, tool_name: str, tool_input: dict,
                  function_name: str = None, round_number: int = 1):
        """Emit a tool call event — visible in the agent feed."""
        self._emit(PipelineEvent(
            event_type=EventType.TOOL_CALL,
            agent=agent,
            message=f"Calling {tool_name}",
            function_name=function_name,
            round_number=round_number,
            data={"tool": tool_name, "input_keys": list(tool_input.keys())},
        ))

    def tool_result(self, agent: str, tool_name: str, summary: str,
                    function_name: str = None, round_number: int = 1):
        """Emit a tool result event."""
        self._emit(PipelineEvent(
            event_type=EventType.TOOL_RESULT,
            agent=agent,
            message=f"{tool_name}: {summary}",
            function_name=function_name,
            round_number=round_number,
            data={"tool": tool_name},
        ))

    def thinking(self, agent: str, thinking_text: str,
                 function_name: str = None, round_number: int = 1):
        """Emit a thinking trace — shows Claude's reasoning in the feed."""
        preview = thinking_text[:300] + "..." if len(thinking_text) > 300 else thinking_text
        self._emit(PipelineEvent(
            event_type=EventType.THINKING,
            agent=agent,
            message=preview,
            function_name=function_name,
            round_number=round_number,
            data={"full_length": len(thinking_text)},
        ))