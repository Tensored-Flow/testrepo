"""Agent 1: Triage — repo scanner and function classifier.

Scans a repository, finds all Python functions, and classifies each as:
  Category A: Pure, benchmarkable (no side effects)
  Category B: Has side effects (DB, file I/O, network, GPU) — analysis only
  Category C: Skip (too trivial, test, config, etc.)

This agent does NOT use the LLM. It's pure AST analysis + heuristics.
"""

from __future__ import annotations

import ast
import json
import os
from typing import Any

import lizard


# ── Constants ────────────────────────────────────────────────

SKIP_DIRS = {
    "__pycache__", ".git", "venv", "env", "node_modules", "migrations",
    ".eggs", ".tox", "build", "dist", ".mypy_cache", ".pytest_cache",
}

SKIP_FILENAMES = {
    "setup.py", "setup.cfg", "conftest.py", "manage.py", "wsgi.py", "asgi.py",
}

DUNDER_METHODS = {
    "__init__", "__repr__", "__str__", "__eq__", "__ne__", "__lt__", "__le__",
    "__gt__", "__ge__", "__hash__", "__bool__", "__len__", "__contains__",
    "__getitem__", "__setitem__", "__delitem__", "__iter__", "__next__",
    "__enter__", "__exit__", "__call__", "__del__", "__new__", "__format__",
    "__bytes__", "__sizeof__", "__reduce__", "__reduce_ex__", "__getattr__",
    "__setattr__", "__delattr__", "__get__", "__set__", "__delete__",
    "__init_subclass__", "__class_getitem__", "__missing__", "__reversed__",
    "__add__", "__radd__", "__iadd__", "__sub__", "__rsub__", "__isub__",
    "__mul__", "__rmul__", "__imul__", "__truediv__", "__floordiv__",
    "__mod__", "__pow__", "__and__", "__or__", "__xor__", "__neg__",
    "__pos__", "__abs__", "__invert__", "__complex__", "__int__", "__float__",
    "__index__", "__round__", "__trunc__", "__floor__", "__ceil__",
    "__aenter__", "__aexit__", "__aiter__", "__anext__",
}

RED_FLAG_NAMES = {
    "requests", "urllib", "http", "httpx", "aiohttp", "socket", "grpc",
    "websocket", "sqlite3", "sqlalchemy", "pymongo", "redis", "psycopg2",
    "cursor", "torch", "tensorflow", "tf", "keras", "cuda",
    "subprocess", "shutil", "tempfile",
}

RED_FLAG_ATTRS = {
    "requests.get", "requests.post", "requests.put", "requests.delete",
    "requests.patch", "requests.head", "requests.session",
    "urllib.request", "urllib.urlopen",
    "session.query", "session.execute", "session.commit", "session.rollback",
    "cursor.execute", "cursor.fetchone", "cursor.fetchall",
    "os.system", "os.popen", "os.exec", "os.execl", "os.execv",
    "os.execvp", "os.execvpe",
    "torch.cuda", "model.train", "model.eval", "model.to",
    "Path.write_text", "Path.write_bytes", "Path.read_text", "Path.read_bytes",
}

RED_FLAG_CALLS = {"open", "print", "exec", "eval", "input"}

SIDE_EFFECT_PARAM_NAMES = {
    "session", "request", "response", "conn", "connection", "cursor",
    "db", "client", "socket", "app", "config",
}

SIDE_EFFECT_PARAM_TYPES = {
    "Session", "Request", "Response", "Connection", "Cursor",
    "Client", "Socket", "Flask", "FastAPI",
}


# ── Event Emitter ────────────────────────────────────────────

def emit(agent: str, message: str) -> None:
    print(json.dumps({"agent": agent, "message": message}))


# ── AST Side-Effect Detector ─────────────────────────────────

class SideEffectDetector(ast.NodeVisitor):
    """Walk a function's AST subtree and collect red flags for side effects."""

    def __init__(self, file_imports: set[str]):
        self.red_flags: list[str] = []
        self.file_imports = file_imports

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in RED_FLAG_NAMES:
            self.red_flags.append(f"Reference to '{node.id}'")
        if node.id in self.file_imports and node.id in RED_FLAG_NAMES:
            self.red_flags.append(f"Uses imported module '{node.id}'")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        chain = _get_attr_chain(node)
        for flag in RED_FLAG_ATTRS:
            if flag in chain:
                self.red_flags.append(f"Attribute access: {chain}")
                break
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func_name = _get_call_name(node)
        if func_name in RED_FLAG_CALLS:
            self.red_flags.append(f"Call to {func_name}()")
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        self.red_flags.append(f"Uses global: {node.names}")
        self.generic_visit(node)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.red_flags.append(f"Uses nonlocal: {node.names}")
        self.generic_visit(node)


def _get_attr_chain(node: ast.Attribute) -> str:
    """Reconstruct dotted attribute chain like 'requests.get'."""
    parts = [node.attr]
    current = node.value
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _get_call_name(node: ast.Call) -> str:
    """Get the name of a function being called."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


# ── File-Level Helpers ───────────────────────────────────────

def _extract_file_imports(tree: ast.Module) -> set[str]:
    """Extract all imported module names from a module-level AST."""
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
    return imports


def _is_skip_file(rel_path: str, source: str) -> tuple[bool, str]:
    """Check if a file should be skipped entirely (Category C at file level)."""
    basename = os.path.basename(rel_path)
    parts = rel_path.replace("\\", "/").split("/")

    # Files inside tests/ or test/ directories
    for part in parts[:-1]:
        if part in ("tests", "test"):
            return True, f"File in {part}/ directory"

    # Known skip filenames
    if basename in SKIP_FILENAMES:
        return True, f"Config/setup file: {basename}"

    # Files starting with test_
    if basename.startswith("test_"):
        return True, f"Test file: {basename}"

    # __init__.py that is empty or only has imports
    if basename == "__init__.py":
        stripped = source.strip()
        if not stripped:
            return True, "Empty __init__.py"
        try:
            tree = ast.parse(source)
            has_non_import = False
            for node in ast.iter_child_nodes(tree):
                if not isinstance(node, (ast.Import, ast.ImportFrom, ast.Expr)):
                    has_non_import = True
                    break
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                    continue  # docstring
                if isinstance(node, ast.Expr):
                    has_non_import = True
                    break
            if not has_non_import:
                return True, "__init__.py with only imports"
        except SyntaxError:
            return True, "Unparseable __init__.py"

    return False, ""


def _get_decorator_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Extract decorator names from a function node."""
    names = []
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name):
            names.append(dec.id)
        elif isinstance(dec, ast.Attribute):
            names.append(_get_attr_chain(dec))
        elif isinstance(dec, ast.Call):
            if isinstance(dec.func, ast.Name):
                names.append(dec.func.id)
            elif isinstance(dec.func, ast.Attribute):
                names.append(_get_attr_chain(dec.func))
    return names


def _is_body_trivial(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check if function body is just 'pass', '...', or a single docstring."""
    body = func_node.body
    if len(body) == 1:
        stmt = body[0]
        # pass
        if isinstance(stmt, ast.Pass):
            return True
        # ... (Ellipsis)
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            if stmt.value.value is Ellipsis or isinstance(stmt.value.value, str):
                return True
    if len(body) == 2:
        # docstring + pass/...
        first, second = body
        is_docstring = isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)
        is_trivial = isinstance(second, ast.Pass) or (isinstance(second, ast.Expr) and isinstance(second.value, ast.Constant) and second.value.value is Ellipsis)
        if is_docstring and is_trivial:
            return True
    return False


# ── Function Classification ──────────────────────────────────

def _classify_function(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    file_imports: set[str],
) -> tuple[str, str, list[str]]:
    """Classify a function as A, B, or C.

    Returns (category, reason, red_flags).
    """
    name = func_node.name
    decorators = _get_decorator_names(func_node)
    params = [arg.arg for arg in func_node.args.args]

    # ── Category C checks ──
    if name.startswith("test_"):
        return "C", "Test function", []

    if "pytest.fixture" in decorators or "fixture" in decorators:
        return "C", "pytest fixture", []

    route_decorators = {"app.route", "router.get", "router.post", "router.put",
                        "router.delete", "router.patch", "app.get", "app.post",
                        "app.put", "app.delete", "app.patch"}
    for dec in decorators:
        if any(dec.startswith(rd.split(".")[0]) and rd.split(".")[1] in dec for rd in route_decorators if "." in rd):
            return "C", f"Route/endpoint handler ({dec})", []

    if name in DUNDER_METHODS:
        return "C", f"Dunder method: {name}", []

    if _is_body_trivial(func_node):
        return "C", "Trivial body (pass/... only)", []

    if "property" in decorators:
        return "C", "Property accessor", []

    # ── Category B checks ──
    is_async = isinstance(func_node, ast.AsyncFunctionDef)
    red_flags: list[str] = []

    if is_async:
        red_flags.append("Async function")

    # Check parameter names for side-effect hints
    for param in params:
        if param in SIDE_EFFECT_PARAM_NAMES:
            red_flags.append(f"Parameter '{param}' suggests external dependency")

    # Check parameter type annotations
    for arg in func_node.args.args:
        if arg.annotation:
            ann_name = ""
            if isinstance(arg.annotation, ast.Name):
                ann_name = arg.annotation.id
            elif isinstance(arg.annotation, ast.Attribute):
                ann_name = arg.annotation.attr
            if ann_name in SIDE_EFFECT_PARAM_TYPES:
                red_flags.append(f"Parameter typed as '{ann_name}'")

    # Walk function body for side effects
    detector = SideEffectDetector(file_imports)
    for child in func_node.body:
        detector.visit(child)
    red_flags.extend(detector.red_flags)

    # De-duplicate red flags
    red_flags = list(dict.fromkeys(red_flags))

    if red_flags:
        return "B", red_flags[0], red_flags

    # ── Default: Category A ──
    return "A", "Pure function, no side effects detected", []


# ── Test Framework Detection ─────────────────────────────────

def _detect_test_framework(repo_path: str) -> tuple[str | None, str | None]:
    """Detect test framework and test directory."""
    framework = None
    test_dir = None

    # Check for test directories
    for candidate in ["tests", "test"]:
        candidate_path = os.path.join(repo_path, candidate)
        if os.path.isdir(candidate_path):
            test_dir = candidate
            break

    # If no top-level test dir, look deeper
    if test_dir is None:
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            basename = os.path.basename(root)
            if basename in ("tests", "test"):
                test_dir = os.path.relpath(root, repo_path)
                break
            for f in files:
                if f.startswith("test_") and f.endswith(".py"):
                    test_dir = os.path.relpath(root, repo_path)
                    break
            if test_dir:
                break

    # Check for conftest.py
    if os.path.exists(os.path.join(repo_path, "conftest.py")):
        framework = "pytest"

    # Check requirements files for pytest
    for req_file in ["requirements.txt", "requirements-dev.txt", "requirements_dev.txt"]:
        req_path = os.path.join(repo_path, req_file)
        if os.path.exists(req_path):
            try:
                with open(req_path, "r") as f:
                    content = f.read().lower()
                if "pytest" in content:
                    framework = "pytest"
                elif "unittest" in content and framework is None:
                    framework = "unittest"
            except (IOError, UnicodeDecodeError):
                pass

    # Check setup.cfg / pyproject.toml
    for cfg_file in ["setup.cfg", "pyproject.toml"]:
        cfg_path = os.path.join(repo_path, cfg_file)
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r") as f:
                    content = f.read().lower()
                if "pytest" in content:
                    framework = "pytest"
            except (IOError, UnicodeDecodeError):
                pass

    # Scan test files for unittest imports
    if framework is None and test_dir:
        test_abs = os.path.join(repo_path, test_dir)
        if os.path.isdir(test_abs):
            for fname in os.listdir(test_abs):
                if fname.endswith(".py"):
                    try:
                        with open(os.path.join(test_abs, fname), "r") as f:
                            content = f.read()
                        if "import unittest" in content or "from unittest" in content:
                            framework = "unittest"
                            break
                        if "import pytest" in content or "from pytest" in content:
                            framework = "pytest"
                            break
                    except (IOError, UnicodeDecodeError):
                        pass

    return framework, test_dir


# ── Main Agent ───────────────────────────────────────────────

def triage_agent(repo_path: str) -> dict[str, Any]:
    """Scan a Python repository and classify all functions.

    Returns a dict with targets (Category A), analysis_only (B), skipped (C).
    """
    emit("triage", f"Scanning {repo_path}...")

    repo_path = os.path.abspath(repo_path)

    # ── STEP 1: Walk the file tree ──
    py_files: list[str] = []  # absolute paths
    dirs_skipped = 0

    for root, dirs, files in os.walk(repo_path):
        # Filter directories in-place
        original_count = len(dirs)
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        dirs_skipped += original_count - len(dirs)

        for fname in files:
            if fname.endswith(".py"):
                py_files.append(os.path.join(root, fname))

    emit("triage", f"Found {len(py_files)} Python files, skipping {dirs_skipped} directories")

    # ── STEP 2: Classify files, STEP 3: Parse, STEP 4: Classify functions ──
    targets: list[dict] = []        # Category A
    analysis_only: list[dict] = []  # Category B
    skipped: list[dict] = []        # Category C
    files_scanned = 0
    files_skipped = 0
    total_functions = 0

    for filepath in sorted(py_files):
        rel_path = os.path.relpath(filepath, repo_path)

        # Read file
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                source = f.read()
        except (IOError, OSError):
            files_skipped += 1
            continue

        # File-level skip check
        should_skip, skip_reason = _is_skip_file(rel_path, source)
        if should_skip:
            files_skipped += 1
            skipped.append({
                "function_name": "(file)",
                "file_path": rel_path,
                "category": "C",
                "reason": skip_reason,
            })
            continue

        # Parse AST
        try:
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            files_skipped += 1
            continue

        files_scanned += 1
        source_lines = source.splitlines()
        file_imports = _extract_file_imports(tree)

        emit("triage", f"Parsing {rel_path}...")

        # Extract and classify functions
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            total_functions += 1
            func_name = node.name
            start_line = node.lineno
            end_line = node.end_lineno or start_line
            func_source = "\n".join(source_lines[start_line - 1 : end_line])
            params = [arg.arg for arg in node.args.args]

            category, reason, red_flags = _classify_function(
                node, file_imports
            )

            emit("triage", f"  {func_name}() → Category {category}: {reason}")

            if category == "A":
                targets.append({
                    "function_name": func_name,
                    "file_path": rel_path,
                    "source_code": func_source,
                    "start_line": start_line,
                    "end_line": end_line,
                    "cyclomatic_complexity": 0,  # filled in step 5
                    "parameters": params,
                    "category": "A",
                    "reason": reason,
                })
            elif category == "B":
                analysis_only.append({
                    "function_name": func_name,
                    "file_path": rel_path,
                    "source_code": func_source,
                    "category": "B",
                    "reason": reason,
                    "red_flags": red_flags,
                })
            else:
                skipped.append({
                    "function_name": func_name,
                    "file_path": rel_path,
                    "category": "C",
                    "reason": reason,
                })

    # ── STEP 5: Run lizard on Category A functions ──
    emit("triage", f"Running lizard on {len(targets)} Category A functions...")

    # Cache lizard analysis per file
    lizard_cache: dict[str, Any] = {}
    for target in targets:
        abs_path = os.path.join(repo_path, target["file_path"])
        if abs_path not in lizard_cache:
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    file_source = f.read()
                lizard_cache[abs_path] = lizard.analyze_file.analyze_source_code(
                    abs_path, file_source
                )
            except Exception:
                lizard_cache[abs_path] = None

        analysis = lizard_cache.get(abs_path)
        if analysis:
            func_name = target["function_name"]
            for func in analysis.function_list:
                if func.name == func_name or func.name.endswith(f".{func_name}"):
                    target["cyclomatic_complexity"] = func.cyclomatic_complexity
                    break

    # Sort Category A by cyclomatic complexity descending
    targets.sort(key=lambda t: t["cyclomatic_complexity"], reverse=True)

    # ── STEP 6: Detect test framework ──
    framework, test_dir = _detect_test_framework(repo_path)
    emit("triage", f"Detected test framework: {framework}")

    # ── STEP 7: Build result ──
    cat_a = len(targets)
    cat_b = len(analysis_only)
    cat_c = len(skipped)

    emit(
        "triage",
        f"Triage complete: {cat_a} optimizable, {cat_b} analysis-only, {cat_c} skipped",
    )

    return {
        "targets": targets,
        "analysis_only": analysis_only,
        "skipped": skipped,
        "test_framework": framework,
        "test_directory": test_dir,
        "summary": {
            "total_functions": total_functions,
            "category_a": cat_a,
            "category_b": cat_b,
            "category_c": cat_c,
            "files_scanned": files_scanned,
            "files_skipped": files_skipped,
        },
    }
