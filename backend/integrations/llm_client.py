"""
Claude API Client — Full-featured wrapper for agentic tool use.

Supports:
  - Basic text completion (call_claude)
  - JSON mode (call_claude_json)
  - AGENTIC TOOL LOOP (call_claude_with_tools) ← THE KEY FEATURE
  - Extended thinking (budget_tokens for visible reasoning)
  - Streaming (for real-time SSE to frontend)
  - Multi-model (Haiku for fast decisions, Sonnet for reasoning)

The agentic tool loop is what makes this genuinely agentic:
  1. Send prompt + tool definitions to Claude
  2. Claude DECIDES which tools to call (or none)
  3. We execute the tools deterministically
  4. Send results back to Claude
  5. Claude reasons about results, may call more tools
  6. Repeat until Claude gives a final answer
"""

import json
import time
import os
from typing import Optional, Generator

try:
    import anthropic
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "anthropic", "--break-system-packages"])
    import anthropic


# ─── Model Configuration ───
SONNET = "claude-sonnet-4-20250514"
HAIKU = "claude-haiku-4-5-20251001"

DEFAULT_MODEL = SONNET
MAX_TOKENS = 8192
TEMPERATURE = 0.2
MAX_RETRIES = 3
THINKING_BUDGET = 5000  # tokens for extended thinking


def _get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    return anthropic.Anthropic(api_key=api_key)


# ═══════════════════════════════════════════════════════════
# Basic Completion (unchanged from before)
# ═══════════════════════════════════════════════════════════

def call_claude(
    system_prompt: str,
    user_message: str,
    model: str = DEFAULT_MODEL,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
) -> str:
    """Simple text completion. No tools, no thinking."""
    client = _get_client()

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text
        except anthropic.RateLimitError:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                raise
        except anthropic.APIError as e:
            if attempt < MAX_RETRIES - 1 and getattr(e, 'status_code', 500) >= 500:
                time.sleep(1)
            else:
                raise
    return ""


def call_claude_json(
    system_prompt: str,
    user_message: str,
    model: str = DEFAULT_MODEL,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
) -> dict:
    """Completion with JSON parsing + auto-fix."""
    full_system = system_prompt + "\n\nRespond ONLY with valid JSON. No markdown fences, no explanation."
    text = call_claude(full_system, user_message, model, temperature, max_tokens)

    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Could not parse JSON:\n{text[:500]}")


# ═══════════════════════════════════════════════════════════
# AGENTIC TOOL LOOP — This is what wins the track
# ═══════════════════════════════════════════════════════════

def call_claude_with_tools(
    system_prompt: str,
    user_message: str,
    tools: list[dict],
    tool_executor,
    model: str = DEFAULT_MODEL,
    max_turns: int = 15,
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
    enable_thinking: bool = True,
    thinking_budget: int = THINKING_BUDGET,
    on_tool_call=None,
    on_thinking=None,
    on_text=None,
) -> dict:
    """
    THE AGENTIC LOOP.

    Claude autonomously decides which tools to call, interprets results,
    calls more tools if needed, and eventually produces a final answer.

    Args:
        system_prompt: System context for the agent
        user_message: The task/question
        tools: List of tool schemas (Claude tool_use format)
        tool_executor: Function(tool_name, tool_input) → str (JSON result)
        model: Which Claude model to use
        max_turns: Max tool-use rounds before forcing a final answer
        enable_thinking: Whether to enable extended thinking
        thinking_budget: Token budget for thinking
        on_tool_call: Callback(tool_name, tool_input, tool_result) for events
        on_thinking: Callback(thinking_text) for streaming thinking traces
        on_text: Callback(text) for streaming text output

    Returns:
        {
            "response": str,            # Final text answer
            "tool_calls": list[dict],   # All tool calls made
            "thinking": list[str],      # All thinking traces
            "turns": int,               # How many turns it took
            "model": str,               # Which model was used
        }
    """
    client = _get_client()

    messages = [{"role": "user", "content": user_message}]
    all_tool_calls = []
    all_thinking = []
    final_text = ""

    # Build create kwargs
    create_kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system_prompt,
        "tools": tools,
        "messages": messages,
    }

    # Extended thinking
    if enable_thinking:
        try:
            # Test if SDK supports thinking
            create_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
            # Extended thinking requires temperature = 1
            create_kwargs["temperature"] = 1
        except Exception:
            enable_thinking = False

    for turn in range(max_turns):
        for attempt in range(MAX_RETRIES):
            try:
                response = client.messages.create(**create_kwargs)
                break
            except anthropic.RateLimitError:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
            except anthropic.APIError as e:
                if attempt < MAX_RETRIES - 1 and getattr(e, 'status_code', 500) >= 500:
                    time.sleep(1)
                else:
                    raise

        # Process response content blocks
        assistant_content = response.content
        tool_use_blocks = []

        for block in assistant_content:
            block_type = getattr(block, 'type', None)
            if block_type == "thinking":
                thinking_text = block.thinking
                all_thinking.append(thinking_text)
                if on_thinking:
                    on_thinking(thinking_text)

            elif block_type == "text":
                final_text += block.text
                if on_text:
                    on_text(block.text)

            elif block_type == "tool_use":
                tool_use_blocks.append(block)

        # If Claude wants to use tools
        if response.stop_reason == "tool_use" and tool_use_blocks:
            # Add assistant message to conversation
            messages.append({"role": "assistant", "content": assistant_content})

            # Execute each tool and collect results
            tool_results = []
            for tool_block in tool_use_blocks:
                tool_name = tool_block.name
                tool_input = tool_block.input
                tool_id = tool_block.id

                # Execute the tool deterministically
                result_str = tool_executor(tool_name, tool_input)

                tool_call_record = {
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "tool_result": result_str,
                    "turn": turn + 1,
                }
                all_tool_calls.append(tool_call_record)

                if on_tool_call:
                    on_tool_call(tool_name, tool_input, result_str)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result_str,
                })

            # Add tool results to conversation
            messages.append({"role": "user", "content": tool_results})

            # Update create_kwargs with new messages
            create_kwargs["messages"] = messages

        else:
            # Claude gave a final answer (stop_reason == "end_turn")
            break

    return {
        "response": final_text,
        "tool_calls": all_tool_calls,
        "thinking": all_thinking,
        "turns": turn + 1,
        "model": model,
    }


# ═══════════════════════════════════════════════════════════
# STREAMING AGENTIC LOOP — For real-time SSE
# ═══════════════════════════════════════════════════════════

def stream_claude_with_tools(
    system_prompt: str,
    user_message: str,
    tools: list[dict],
    tool_executor,
    model: str = DEFAULT_MODEL,
    max_turns: int = 15,
    max_tokens: int = MAX_TOKENS,
    enable_thinking: bool = True,
    thinking_budget: int = THINKING_BUDGET,
) -> Generator[dict, None, None]:
    """
    Streaming version of the agentic loop.
    Yields events as they happen for real-time SSE streaming.

    Yields dicts with "type" field:
      - {"type": "thinking", "text": "..."}
      - {"type": "text", "text": "..."}
      - {"type": "tool_call", "name": "...", "input": {...}}
      - {"type": "tool_result", "name": "...", "result": "..."}
      - {"type": "done", "response": "...", "tool_calls": [...]}
    """
    client = _get_client()

    messages = [{"role": "user", "content": user_message}]
    all_tool_calls = []
    final_text = ""

    create_kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "tools": tools,
        "messages": messages,
    }

    if enable_thinking:
        create_kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget,
        }
        create_kwargs["temperature"] = 1
    else:
        create_kwargs["temperature"] = TEMPERATURE

    for turn in range(max_turns):
        # Use streaming API
        with client.messages.stream(**create_kwargs) as stream:
            response = stream.get_final_message()

        # Process blocks
        tool_use_blocks = []
        for block in response.content:
            if block.type == "thinking":
                yield {"type": "thinking", "text": block.thinking, "turn": turn + 1}
            elif block.type == "text":
                final_text += block.text
                yield {"type": "text", "text": block.text, "turn": turn + 1}
            elif block.type == "tool_use":
                tool_use_blocks.append(block)
                yield {"type": "tool_call", "name": block.name, "input": block.input, "turn": turn + 1}

        if response.stop_reason == "tool_use" and tool_use_blocks:
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for tool_block in tool_use_blocks:
                result_str = tool_executor(tool_block.name, tool_block.input)

                all_tool_calls.append({
                    "tool_name": tool_block.name,
                    "tool_input": tool_block.input,
                    "tool_result": result_str,
                    "turn": turn + 1,
                })

                yield {
                    "type": "tool_result",
                    "name": tool_block.name,
                    "result": result_str,
                    "turn": turn + 1,
                }

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": result_str,
                })

            messages.append({"role": "user", "content": tool_results})
            create_kwargs["messages"] = messages
        else:
            break

    yield {
        "type": "done",
        "response": final_text,
        "tool_calls": all_tool_calls,
        "turns": turn + 1,
    }


# ═══════════════════════════════════════════════════════════
# Multi-model helpers
# ═══════════════════════════════════════════════════════════

def call_haiku(system_prompt: str, user_message: str, **kwargs) -> str:
    """Quick classification/decision using Haiku (fast + cheap)."""
    return call_claude(system_prompt, user_message, model=HAIKU, **kwargs)


def call_haiku_json(system_prompt: str, user_message: str, **kwargs) -> dict:
    """Quick structured output using Haiku."""
    return call_claude_json(system_prompt, user_message, model=HAIKU, **kwargs)


def call_sonnet(system_prompt: str, user_message: str, **kwargs) -> str:
    """Deep reasoning using Sonnet."""
    return call_claude(system_prompt, user_message, model=SONNET, **kwargs)