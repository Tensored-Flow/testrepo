"""
Contract tests that ANY sandbox implementation must pass.
Run these against both SubprocessSandbox and DockerSandbox.

Usage:
    SANDBOX_BACKEND=subprocess python -m pytest backend/tests/test_sandbox_contract.py
    SANDBOX_BACKEND=docker python -m pytest backend/tests/test_sandbox_contract.py
"""

import pytest
import asyncio
from backend.core.sandbox_factory import create_sandbox


@pytest.fixture
def sandbox():
    return create_sandbox()


@pytest.mark.asyncio
async def test_health_check(sandbox):
    assert await sandbox.health_check()


@pytest.mark.asyncio
async def test_simple_execution(sandbox):
    result = await sandbox.execute("print('hello world')")
    assert result.success
    assert result.stdout.strip() == "hello world"
    assert not result.timed_out


@pytest.mark.asyncio
async def test_timeout_enforcement(sandbox):
    result = await sandbox.execute("import time; time.sleep(30)", timeout=2.0)
    assert not result.success
    assert result.timed_out


@pytest.mark.asyncio
async def test_syntax_error(sandbox):
    result = await sandbox.execute("def foo(:")
    assert not result.success
    assert result.stderr  # Should contain syntax error


@pytest.mark.asyncio
async def test_runtime_error(sandbox):
    result = await sandbox.execute("raise ValueError('test error')")
    assert not result.success


@pytest.mark.asyncio
async def test_import_numpy(sandbox):
    result = await sandbox.execute(
        "import numpy as np; print(np.array([1,2,3]).sum())"
    )
    assert result.success
    assert result.stdout.strip() == "6"


@pytest.mark.asyncio
async def test_import_lizard(sandbox):
    code = """
import lizard
result = lizard.analyze_file.analyze_source_code("test.py", "def foo():\\n    return 1")
print(len(result.function_list))
"""
    result = await sandbox.execute(code)
    assert result.success


@pytest.mark.asyncio
async def test_import_scipy(sandbox):
    result = await sandbox.execute(
        "from scipy import stats; print(stats.ttest_ind([1,2,3],[4,5,6]).pvalue)"
    )
    assert result.success
    assert float(result.stdout.strip()) > 0


@pytest.mark.asyncio
async def test_memory_tracking(sandbox):
    code = """
import tracemalloc
tracemalloc.start()
x = [0] * 1000000
print(tracemalloc.get_traced_memory()[1])
tracemalloc.stop()
"""
    result = await sandbox.execute(code)
    assert result.success
    assert int(result.stdout.strip()) > 0


@pytest.mark.asyncio
async def test_result_types(sandbox):
    """Verify SandboxResult has all required fields."""
    result = await sandbox.execute("print(1)")
    assert hasattr(result, "success")
    assert hasattr(result, "stdout")
    assert hasattr(result, "stderr")
    assert hasattr(result, "timed_out")
    assert hasattr(result, "error")
    assert isinstance(result.success, bool)
    assert isinstance(result.stdout, str)


@pytest.mark.asyncio
async def test_multiline_output(sandbox):
    result = await sandbox.execute("print('line1')\nprint('line2')\nprint('line3')")
    assert result.success
    lines = result.stdout.strip().split("\n")
    assert len(lines) == 3


@pytest.mark.asyncio
async def test_exit_code_nonzero(sandbox):
    result = await sandbox.execute("import sys; sys.exit(1)")
    assert not result.success
