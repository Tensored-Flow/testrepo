"""
ConversationContext — Shared working memory for the multi-agent system.

Instead of agents passing data forward via return values (pipeline style),
ConversationContext acts as shared memory that any agent can read/write.

Each optimization run on a single function gets its own ConversationContext.
Agents read what they need, do their work, and write their output back —
making inter-agent communication explicit and visible.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime


@dataclass
class AgentMessage:
    """A single message in the inter-agent conversation."""
    sender: str                    # e.g. "analyst", "optimizer", "validator"
    recipient: str                 # e.g. "optimizer", "analyst", "all"
    message_type: str              # "hypothesis", "code_submission", "rejection", "diagnosis", "approval", "test_suite", "plan_directive"
    content: dict[str, Any]        # Structured payload (NOT free text — typed per message_type)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    round_number: int = 1
    confidence: Optional[float] = None  # 0.0–1.0, agent's self-assessed confidence


@dataclass
class ConversationContext:
    """Shared working memory for one function's optimization lifecycle."""
    function_name: str
    original_source: str
    file_path: str
    category: str                  # "A", "B", "C"

    # Accumulated agent messages — the full conversation history
    messages: list[AgentMessage] = field(default_factory=list)

    # Shared state that any agent can read/write
    snapshot_metrics: Optional[dict] = None
    current_hypothesis: Optional[str] = None
    optimized_code: Optional[str] = None
    test_suites: Optional[dict] = None
    validation_result: Optional[dict] = None

    # Round tracking
    current_round: int = 1
    max_rounds: int = 3
    rejection_history: list[dict] = field(default_factory=list)

    # Plan directives from Planner (constraints the agents must respect)
    plan_directives: Optional[dict] = None

    # Optional event emitter reference for auto-emitting handoff events
    _event_emitter: Optional[Any] = field(default=None, repr=False)

    def set_emitter(self, emitter):
        """Attach an EventEmitter to auto-emit AGENT_HANDOFF events."""
        self._event_emitter = emitter

    def add_message(
        self,
        sender: str,
        recipient: str,
        message_type: str,
        content: dict,
        confidence: float = None,
    ) -> AgentMessage:
        msg = AgentMessage(
            sender=sender,
            recipient=recipient,
            message_type=message_type,
            content=content,
            round_number=self.current_round,
            confidence=confidence,
        )
        self.messages.append(msg)

        # Auto-emit AGENT_HANDOFF event if emitter is attached
        if self._event_emitter:
            try:
                from backend.models.events import EventType
                self._event_emitter.complete(
                    EventType.AGENT_HANDOFF,
                    sender,
                    f"{sender} → {recipient}: {message_type}",
                    function_name=self.function_name,
                    round_number=self.current_round,
                    data={
                        "from": sender,
                        "to": recipient,
                        "type": message_type,
                        "summary": _summarize_content(content, max_len=200),
                        "round": self.current_round,
                    },
                )
            except Exception:
                pass  # Don't let event emission break the pipeline

        return msg

    def get_messages_for(self, recipient: str) -> list[AgentMessage]:
        """Get all messages addressed to a specific agent or 'all'."""
        return [m for m in self.messages if m.recipient in (recipient, "all")]

    def get_messages_from(self, sender: str) -> list[AgentMessage]:
        return [m for m in self.messages if m.sender == sender]

    def get_rejection_summary(self) -> str:
        """Compile a natural-language summary of all past rejections for the Analyst's diagnosis prompt."""
        if not self.rejection_history:
            return "No prior rejections."
        parts = []
        for i, rej in enumerate(self.rejection_history, 1):
            vetoes = ", ".join(rej.get("vetoes_triggered", []))
            parts.append(
                f"Round {i}: REJECTED. Vetoes: [{vetoes}]. Details: {rej.get('summary', 'N/A')}"
            )
        return "\n".join(parts)

    def to_agent_briefing(self, agent_name: str) -> str:
        """Generate a context briefing string that gets injected into an agent's system prompt.
        Each agent gets the FULL conversation so far — they can see what every other agent said."""
        lines = [
            f"=== Conversation History for {self.function_name} "
            f"(Round {self.current_round}/{self.max_rounds}) ==="
        ]
        for msg in self.messages:
            lines.append(
                f"[{msg.sender} → {msg.recipient}] ({msg.message_type}): "
                f"{_summarize_content(msg.content, max_len=500)}"
            )
        if self.plan_directives:
            lines.append(f"\n=== Planner Directives ===\n{self.plan_directives}")
        return "\n".join(lines)


def _summarize_content(content: dict, max_len: int = 500) -> str:
    """Truncate large payloads (like full source code) for the briefing string."""
    s = str(content)
    return s[:max_len] + "..." if len(s) > max_len else s
