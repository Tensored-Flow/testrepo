# Module: integrations
# Owner: ___
# Status: IN PROGRESS
# Depends on: (external APIs — Anthropic, GitHub)
#
# External service integrations. LLM client and GitHub PR creation.

from backend.integrations.llm_client import (
    call_claude,
    call_claude_json,
    call_claude_with_tools,
    call_haiku_json,
    SONNET,
)
