"""
Docker-based sandbox implementation.

TEAMMATE: This is your file. Implement the DockerSandbox class below.

Requirements:
- Must subclass SandboxBackend from sandbox_interface.py
- Must enforce timeout at container level (not Python-level)
- Must enforce memory_limit_mb
- Container must have pre-installed: python3, lizard, radon, numpy, scipy
- stdout/stderr must be captured and returned in SandboxResult
- If network_access=False, container must have no network
- Cleanup must kill any orphaned containers

The rest of the system calls sandbox via:
    sandbox = create_sandbox()  # from sandbox_factory.py
    result = await sandbox.execute(code, timeout=10.0)

That's it. Everything flows through SandboxResult.

Test with:
    SANDBOX_BACKEND=docker python -m pytest backend/tests/test_sandbox_contract.py
"""

from backend.core.sandbox_interface import SandboxBackend, SandboxResult


class DockerSandbox(SandboxBackend):

    def __init__(self, image: str = "complexityimprover-sandbox:latest"):
        self.image = image
        # TODO: Initialize Docker client

    async def execute(
        self,
        code: str,
        timeout: float = 10.0,
        memory_limit_mb: int = 256,
        network_access: bool = False,
    ) -> SandboxResult:
        raise NotImplementedError(
            "Docker sandbox not yet implemented — see docstring above"
        )

    async def health_check(self) -> bool:
        raise NotImplementedError

    async def cleanup(self) -> None:
        raise NotImplementedError
