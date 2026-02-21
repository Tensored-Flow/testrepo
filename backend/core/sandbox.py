"""Safe code execution via exec() with timeout and resource limits.

All function execution during benchmarking and testing goes through this
module. Never exec() user code directly — always use run_in_sandbox().

Owner: ___
Status: NOT STARTED
Depends on: (stdlib only)
"""

from __future__ import annotations

import signal
from typing import Any, Callable


class SandboxTimeout(Exception):
    """Raised when sandboxed code exceeds the time limit."""
    pass


class SandboxError(Exception):
    """Raised when sandboxed code throws an exception."""
    pass


def run_in_sandbox(
    source_code: str,
    function_name: str,
    args: tuple = (),
    kwargs: dict | None = None,
    timeout_seconds: int = 30,
) -> Any:
    """Execute a function from source code in a restricted namespace.

    Compiles and exec()s the source, extracts the named function, calls
    it with the given args/kwargs, and returns the result.

    Raises SandboxTimeout if execution exceeds timeout_seconds.
    Raises SandboxError if the code throws.

    Owner: ___
    Status: NOT STARTED
    Depends on: signal (stdlib)
    """
    # TODO: Create restricted globals (no __import__ override, limited builtins)
    # TODO: exec(source_code, restricted_globals)
    # TODO: Extract function_name from the namespace
    # TODO: Set signal.alarm(timeout_seconds) for Unix timeout
    # TODO: Call the function with args/kwargs
    # TODO: Catch and wrap exceptions in SandboxError
    # TODO: Return the result
    pass


def compile_function(
    source_code: str,
    function_name: str,
) -> Callable:
    """Compile source code and extract a callable function.

    Simpler than run_in_sandbox — just returns the function object
    for repeated calls (e.g., in benchmarking loops).

    Owner: ___
    Status: NOT STARTED
    Depends on: (stdlib only)
    """
    # TODO: exec() the source code
    # TODO: Extract and return the named function
    # TODO: Raise if function not found in namespace
    pass
