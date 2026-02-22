"""
Sandbox — Safe code execution with timeout and resource limits.

Provides both:
1. SubprocessSandbox — async SandboxBackend implementation (new interface)
2. Legacy sync functions — run_in_sandbox(), compile_function(), etc.
   These are kept for backward compat with existing agent code and will
   delegate to SubprocessSandbox internally.

YOUR TEAMMATE'S CONTAINERIZATION PLUGS IN HERE.
Replace SubprocessSandbox with DockerSandbox in sandbox_factory.py.
"""

import subprocess
import sys
import textwrap
import tempfile
import os
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Callable

from backend.core.sandbox_interface import (
    SandboxBackend,
    SandboxResult,
)


# ─────────────────────────────────────────────
# SubprocessSandbox — Async SandboxBackend
# ─────────────────────────────────────────────

class SubprocessSandbox(SandboxBackend):
    """Local subprocess-based sandbox. NOT SECURE. For development only."""

    async def execute(
        self,
        code: str,
        timeout: float = 10.0,
        memory_limit_mb: int = 256,
        network_access: bool = False,
    ) -> SandboxResult:
        start = time.perf_counter()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=tempfile.gettempdir()
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tempfile.gettempdir(),
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            return SandboxResult(
                success=(result.returncode == 0),
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
                error=result.stderr if result.returncode != 0 else None,
                execution_time_ms=elapsed_ms,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                success=False,
                stdout="",
                stderr="",
                error=f"Execution timed out after {timeout}s",
                timed_out=True,
            )
        except Exception as e:
            return SandboxResult(
                success=False,
                stdout="",
                stderr=str(e),
                error=str(e),
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    async def health_check(self) -> bool:
        r = await self.execute("print('ok')", timeout=5.0)
        return r.success and r.stdout.strip() == "ok"

    async def cleanup(self) -> None:
        pass  # Nothing to clean up for subprocess


# ─────────────────────────────────────────────
# Legacy sync API (backward compat)
# ─────────────────────────────────────────────

def run_in_sandbox(
    code: str,
    timeout: float = 10.0,
    capture_output: bool = True,
) -> SandboxResult:
    """
    Execute Python code in a subprocess with timeout.
    Sync wrapper — delegates to SubprocessSandbox but runs synchronously.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=tempfile.gettempdir()
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tempfile.gettempdir(),
        )
        return SandboxResult(
            success=(result.returncode == 0),
            stdout=result.stdout,
            stderr=result.stderr,
            error=result.stderr if result.returncode != 0 else None,
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(
            success=False,
            stdout="",
            stderr="",
            error=f"Execution timed out after {timeout}s",
            timed_out=True,
        )
    except Exception as e:
        return SandboxResult(
            success=False,
            stdout="",
            stderr=str(e) if str(e) else "",
            error=str(e),
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ─────────────────────────────────────────────
# Compile a function from source code
# ─────────────────────────────────────────────

def compile_function(source_code: str, function_name: str) -> Optional[Callable]:
    """
    Compile a function from source and return a callable.
    Returns None if compilation fails.
    """
    try:
        namespace = {}
        exec(compile(textwrap.dedent(source_code), "<sandbox>", "exec"), namespace)
        func = namespace.get(function_name)
        if callable(func):
            return func
        return None
    except Exception:
        return None


# ─────────────────────────────────────────────
# Run a function with specific inputs (for snapshot/validator)
# ─────────────────────────────────────────────

def run_function_with_inputs(
    source_code: str,
    function_name: str,
    test_inputs: list[dict],
    timeout: float = 10.0,
) -> list[dict]:
    """
    Execute a function with a list of inputs and capture outputs.
    Each input is a dict like {"args": [...], "kwargs": {...}}.
    Returns list of {"input": ..., "output": ..., "output_type": ..., "error": ...}.
    """
    inputs_json = json.dumps(test_inputs)
    func_code = textwrap.dedent(source_code)
    script = (
        "import json, sys, traceback\n\n"
        + func_code + "\n\n"
        + f"func = {function_name}\n"
        + f"inputs = json.loads('''{inputs_json}''')\n"
        + "results = []\n"
        + "for inp in inputs:\n"
        + "    args = inp.get('args', [])\n"
        + "    kwargs = inp.get('kwargs', {})\n"
        + "    try:\n"
        + "        output = func(*args, **kwargs)\n"
        + "        results.append({\n"
        + "            'input': inp,\n"
        + "            'output': repr(output),\n"
        + "            'output_type': type(output).__name__,\n"
        + "            'error': None,\n"
        + "        })\n"
        + "    except Exception as e:\n"
        + "        results.append({\n"
        + "            'input': inp,\n"
        + "            'output': None,\n"
        + "            'output_type': None,\n"
        + "            'error': f'{type(e).__name__}: {e}',\n"
        + "        })\n"
        + "print(json.dumps(results))\n"
    )

    result = run_in_sandbox(script, timeout=timeout)

    if result.success and result.stdout.strip():
        try:
            return json.loads(result.stdout.strip())
        except json.JSONDecodeError:
            return [{"error": f"JSON parse error: {result.stdout[:200]}"}]

    return [{"error": result.error or "Sandbox execution failed"}]


# ─────────────────────────────────────────────
# Benchmark a function (timing + memory)
# ─────────────────────────────────────────────

def benchmark_function(
    source_code: str,
    function_name: str,
    input_generator_code: str,
    sizes: list[int] = None,
    runs_per_size: int = 5,
    timeout: float = 30.0,
) -> list[dict]:
    """
    Benchmark a function at various input sizes.
    input_generator_code should define: generate_input(n) -> args tuple
    """
    if sizes is None:
        sizes = [100, 500, 1000, 5000]

    sizes_json = json.dumps(sizes)
    func_code = textwrap.dedent(source_code)
    gen_code = textwrap.dedent(input_generator_code)

    script = (
        "import json, time, statistics, tracemalloc\n\n"
        + func_code + "\n\n"
        + gen_code + "\n\n"
        + f"func = {function_name}\n"
        + f"sizes = {sizes_json}\n"
        + f"runs = {runs_per_size}\n"
        + "results = []\n"
        + "for n in sizes:\n"
        + "    try:\n"
        + "        inp = generate_input(n)\n"
        + "        if not isinstance(inp, tuple):\n"
        + "            inp = (inp,)\n"
        + "        # Warmup run (untimed, primes caches/JIT)\n"
        + "        func(*inp)\n"
        + "        # Loop 1: Timing only (no tracemalloc overhead)\n"
        + "        times = []\n"
        + "        timed_out = False\n"
        + "        for _ in range(runs):\n"
        + "            inp = generate_input(n)\n"
        + "            if not isinstance(inp, tuple):\n"
        + "                inp = (inp,)\n"
        + "            start = time.perf_counter()\n"
        + "            func(*inp)\n"
        + "            elapsed = time.perf_counter() - start\n"
        + "            times.append(elapsed)\n"
        + "            if elapsed > 10:\n"
        + "                timed_out = True\n"
        + "                break\n"
        + "        # Loop 2: Memory measurement (separate, not timed)\n"
        + "        inp = generate_input(n)\n"
        + "        if not isinstance(inp, tuple):\n"
        + "            inp = (inp,)\n"
        + "        tracemalloc.start()\n"
        + "        func(*inp)\n"
        + "        _, peak = tracemalloc.get_traced_memory()\n"
        + "        tracemalloc.stop()\n"
        + "        results.append({\n"
        + "            'size': n,\n"
        + "            'mean_time': statistics.mean(times),\n"
        + "            'std_time': statistics.stdev(times) if len(times) > 1 else 0.0,\n"
        + "            'raw_times': times,\n"
        + "            'memory_bytes': peak,\n"
        + "        })\n"
        + "        if timed_out:\n"
        + "            break\n"
        + "    except Exception as e:\n"
        + "        results.append({\n"
        + "            'size': n,\n"
        + "            'mean_time': -1,\n"
        + "            'std_time': 0,\n"
        + "            'raw_times': [],\n"
        + "            'memory_bytes': 0,\n"
        + "            'error': str(e),\n"
        + "        })\n"
        + "print(json.dumps(results))\n"
    )

    result = run_in_sandbox(script, timeout=timeout)

    if result.success and result.stdout.strip():
        try:
            return json.loads(result.stdout.strip())
        except json.JSONDecodeError:
            return []

    return []


# ─────────────────────────────────────────────
# Run pytest/unittest on code
# ─────────────────────────────────────────────

def run_tests(
    test_code: str,
    function_source: str = "",
    timeout: float = 15.0,
) -> SandboxResult:
    """
    Run test code that contains pytest-style test functions.
    Optionally prepends function source so tests can reference it.
    """
    full_code = ""
    if function_source:
        full_code += textwrap.dedent(function_source) + "\n\n"
    full_code += textwrap.dedent(test_code)

    escaped_code = full_code.replace("'''", '"""').strip()
    script = (
        "import sys\n"
        "import tempfile\n"
        "import os\n\n"
        f"code = '''{escaped_code}'''\n\n"
        "with tempfile.NamedTemporaryFile(mode='w', suffix='_test.py', delete=False) as f:\n"
        "    f.write(code)\n"
        "    path = f.name\n\n"
        "try:\n"
        "    import pytest\n"
        "    exit_code = pytest.main([path, '-v', '--tb=short', '--no-header'])\n"
        "    sys.exit(exit_code)\n"
        "except ImportError:\n"
        "    exec(compile(open(path).read(), path, 'exec'))\n"
        "finally:\n"
        "    os.unlink(path)\n"
    )

    return run_in_sandbox(script, timeout=timeout)
