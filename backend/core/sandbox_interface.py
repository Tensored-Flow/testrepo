"""
Abstract Sandbox Interface — The contract for code execution backends.

This defines the ONLY interface through which the rest of the system
executes arbitrary Python code. Any sandbox backend (subprocess, Docker,
nsjail, Firecracker) must implement SandboxBackend.

Integration contract:
  - code in → SandboxResult out
  - Timeout enforced at backend level
  - Memory limit enforced at backend level
  - All required packages pre-installed
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class SandboxResult:
    """The ONLY return type from any sandbox execution. This is the contract."""
    success: bool
    stdout: str
    stderr: str
    error: Optional[str] = None
    timed_out: bool = False
    exit_code: int = 0
    execution_time_ms: Optional[float] = None
    peak_memory_bytes: Optional[int] = None
    # Legacy field preserved for backward compat with existing code
    return_value: object = None
    # Alias for backward compat
    memory_bytes: int = 0

    def __post_init__(self):
        if self.peak_memory_bytes and not self.memory_bytes:
            self.memory_bytes = self.peak_memory_bytes


class SandboxBackend(ABC):
    """Abstract interface for code execution backends.

    The teammate implementing Docker/nsjail should subclass this.
    Everything else in the system calls through this interface.
    """

    @abstractmethod
    async def execute(
        self,
        code: str,
        timeout: float = 10.0,
        memory_limit_mb: int = 256,
        network_access: bool = False,
    ) -> SandboxResult:
        """Execute Python code in isolation.

        Args:
            code: Complete Python source code to execute (must be self-contained)
            timeout: Max execution time in seconds
            memory_limit_mb: Memory limit for the sandbox
            network_access: Whether to allow network access (for web research tool)

        Returns:
            SandboxResult with stdout, stderr, success flag, timing, memory

        IMPORTANT for implementer:
        - code may import: numpy, scipy, collections, itertools, functools, typing, math, time, tracemalloc
        - code will often end with print() statements — stdout is how we get results back
        - The sandbox MUST have lizard, radon, numpy, scipy pre-installed
        - Timeout must be enforced at the container level, not just in Python
        - If timed_out=True, success must be False
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Verify the sandbox backend is operational."""
        ...

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up any lingering containers/processes."""
        ...
