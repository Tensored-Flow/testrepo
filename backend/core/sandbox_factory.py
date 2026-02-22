"""
Sandbox Factory — Returns the appropriate sandbox backend.

Detects available backends and returns the best one:
1. Docker sandbox (if available and healthy)
2. Subprocess sandbox (fallback, not secure, dev only)

Override with SANDBOX_BACKEND=docker|subprocess environment variable.
"""

import os
from backend.core.sandbox_interface import SandboxBackend


def create_sandbox() -> SandboxBackend:
    """Factory that returns the appropriate sandbox backend.

    The teammate's Docker implementation should:
    1. Create a class like `DockerSandbox(SandboxBackend)` in docker_sandbox.py
    2. The detection logic here handles the rest
    3. Return DockerSandbox() when available, SubprocessSandbox() as fallback

    Environment variable override: SANDBOX_BACKEND=docker|subprocess
    """
    backend = os.environ.get("SANDBOX_BACKEND", "auto")

    if backend == "docker":
        try:
            from backend.core.docker_sandbox import DockerSandbox
            return DockerSandbox()
        except ImportError:
            print(
                "WARNING: Docker sandbox requested but not available. "
                "Falling back to subprocess."
            )
            return _make_subprocess_sandbox()

    if backend == "auto":
        # Try Docker first, fall back gracefully
        try:
            from backend.core.docker_sandbox import DockerSandbox
            sandbox = DockerSandbox()
            # Quick health check
            import asyncio
            loop = asyncio.new_event_loop()
            try:
                if loop.run_until_complete(sandbox.health_check()):
                    print("INFO: Using Docker sandbox backend.")
                    return sandbox
            except Exception:
                pass
            finally:
                loop.close()
        except Exception:
            pass
        print("INFO: Using subprocess sandbox backend (not secure, dev only).")

    return _make_subprocess_sandbox()


def _make_subprocess_sandbox() -> SandboxBackend:
    from backend.core.sandbox import SubprocessSandbox
    return SubprocessSandbox()
