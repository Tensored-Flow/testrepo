"""
Shared wrapper for Claude API calls. All agents import from here.

Usage:
    from llm_client import call_claude, call_claude_json

    result = call_claude_json(
        system_prompt="You are a code analyst.",
        user_message=f"Analyze this function:\n{source_code}"
    )
"""

import json
import os
import time

import anthropic

CLIENT = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096
TEMPERATURE = 0.2
MAX_RETRIES = 3


def call_claude(system_prompt: str, user_message: str, *, json_mode: bool = False) -> str:
    """Call Claude and return the text response. Retries on rate limits."""
    if json_mode:
        system_prompt = system_prompt + "\n\nRespond ONLY with valid JSON, no markdown."

    for attempt in range(MAX_RETRIES):
        try:
            response = CLIENT.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            text = response.content[0].text

            # Log agent name (caller) + truncated response
            import inspect
            caller = inspect.stack()[1].filename.split("/")[-1].replace(".py", "")
            print(f"[{caller}] {text[:100]}")

            return text

        except anthropic.RateLimitError:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"Rate limited, retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
            else:
                raise


def call_claude_json(system_prompt: str, user_message: str) -> dict:
    """Call Claude expecting JSON back. Retries once if parsing fails."""
    text = call_claude(system_prompt, user_message, json_mode=True)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Ask Claude to fix its own output
        fix_prompt = "The following text was supposed to be valid JSON but isn't. Return ONLY the corrected valid JSON, nothing else."
        fixed = call_claude(fix_prompt, text, json_mode=True)
        return json.loads(fixed)
