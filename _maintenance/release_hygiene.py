"""Dependency-free release hygiene checks for the standalone companion tree."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys


TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".css",
    ".cu",
    ".cuh",
    ".html",
    ".ini",
    ".json",
    ".js",
    ".key",
    ".lock",
    ".md",
    ".mjs",
    ".pem",
    ".py",
    ".sh",
    ".svg",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
TEXT_NAMES = {"Makefile", ".gitignore"}
SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "venv",
}
GENERATED_PARTS = {
    ".DS_Store",
    ".ipynb_checkpoints",
}
GENERATED_SUFFIXES = {".pyc", ".pyo"}
LOCAL_PATH_RE = re.compile(
    "(?:file:" + "//|(?<![A-Za-z0-9._-])(?:/" + "Users/|/pri" + "vate/|"
    "/ho"
    + "me/[A-Za-z0-9._-]+/|/"
    + "Applications/|/"
    + "Volumes/|/opt/"
    + "homebrew/|/usr/"
    + "local/)|"
    + r"[A-Za-z]:[\\/]+(?:Users|Program Files(?: \(x86\))?)[\\/]+)"
)
# Keep regression fixtures and these patterns free of literal local identifiers.
LOCAL_MACHINE_RE = re.compile(
    r"(?<![A-Za-z0-9])" + "pan" + "ther" + r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
MACHINE_IDENTITY_RE = re.compile(
    r"[\"'](?:hostname|host_name|machine_name|username|user_name)[\"']"
    r"\s*:\s*[\"'][^\"']+[\"']",
    re.IGNORECASE,
)
SECRET_RE = re.compile(
    r"\b(?:HF_TOKEN|OPENAI_API_KEY|ANTHROPIC_API_KEY|AWS_SECRET_ACCESS_KEY|"
    r"GITHUB_TOKEN|WANDB_API_KEY)\s*=",
    re.IGNORECASE,
)
PRIVATE_KEY_RE = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")
README_PLACEHOLDER_RE = re.compile(
    r"\b(?:to be announced|coming soon|TBD|FIXME)\b",
    re.IGNORECASE,
)
README_CODE_MODULE_RE = re.compile(
    r"(?m)^[ \t]*(?:python(?:3(?:\.\d+)?)?[^\n]*)?"
    r"-m[ \t]+code(?:\.|[ \t]*$)"
)
README_DIRECT_TEST_RE = re.compile(
    r"(?m)^[ \t]*python(?:3(?:\.\d+)?)?"
    r"(?:[ \t]+-[A-Za-z]+)*"
    r"(?:[ \t]+|[ \t]*\\[ \t]*\n[ \t]*)"
    r"(?:code/)?[A-Za-z0-9_]+/test_[A-Za-z0-9_]+\.py[ \t]*$"
)
NOTEBOOK_SHELL_RE = re.compile(
    r"^\s*(?:!\s*(?:python|pytest|pip|bash|sh)\b|%{1,2}(?:pip|bash|sh)\b)",
    re.MULTILINE,
)


def public_definition_violations(path: Path, text: str) -> list[Violation]:
    """Return missing-docstring findings for a companion module's public API."""

    if (
        path.suffix != ".py"
        or path.name.startswith("test_")
        or path.parent.name == "_maintenance"
    ):
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    public_types = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    return [
        Violation(
            path,
            node.lineno,
            f"public definition {node.name} needs a docstring",
        )
        for node in tree.body
        if isinstance(node, public_types)
        and not node.name.startswith("_")
        and ast.get_docstring(node, clean=False) is None
    ]


@dataclass(frozen=True, order=True)
class Violation:
    """One release-blocking finding."""

    path: Path
    line: int
    message: str


def discover_files(root: Path) -> list[Path]:
    """Return every releasable file or symlink without requiring Git."""

    return sorted(
        path
        for path in root.rglob("*")
        if SKIP_DIRS.isdisjoint(path.relative_to(root).parts)
        and (path.is_file() or path.is_symlink())
    )


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def audit_text(path: Path, text: str) -> list[Violation]:
    findings: list[Violation] = []
    for pattern, message in (
        (LOCAL_PATH_RE, "file leaks a machine-local absolute path"),
        (LOCAL_MACHINE_RE, "file leaks a known local machine name"),
        (MACHINE_IDENTITY_RE, "file contains machine or user identity metadata"),
        (SECRET_RE, "file appears to assign a secret-like variable"),
        (PRIVATE_KEY_RE, "file contains a private-key header"),
    ):
        findings.extend(
            Violation(path, line_number(text, match.start()), message)
            for match in pattern.finditer(text)
        )
    if path.name.lower() == "readme.md":
        findings.extend(
            Violation(
                path,
                line_number(text, match.start()),
                "README contains release-placeholder wording",
            )
            for match in README_PLACEHOLDER_RE.finditer(text)
        )
        findings.extend(
            Violation(
                path,
                line_number(text, match.start()),
                "README invokes an unresolvable 'python -m code...' module path",
            )
            for match in README_CODE_MODULE_RE.finditer(text)
        )
        findings.extend(
            Violation(
                path,
                line_number(text, match.start()),
                "README executes a package test as a file; use the companion test runner",
            )
            for match in README_DIRECT_TEST_RE.finditer(text)
        )
    findings.extend(public_definition_violations(path, text))
    return findings


def notebook_source(cell: dict[str, object]) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    return source if isinstance(source, str) else ""


def audit_notebook(path: Path) -> list[Violation]:
    """Reject invalid, executed, output-bearing, or shell-running notebooks."""

    try:
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        line = error.lineno if isinstance(error, json.JSONDecodeError) else 1
        return [Violation(path, line, "notebook is not valid readable JSON")]
    if not isinstance(payload, dict) or not isinstance(payload.get("cells"), list):
        return [Violation(path, 1, "notebook root must contain a cells list")]
    findings: list[Violation] = []
    sources: list[str] = []
    for index, raw_cell in enumerate(payload["cells"], start=1):
        if not isinstance(raw_cell, dict):
            findings.append(Violation(path, index, "notebook cell is not an object"))
            continue
        source = notebook_source(raw_cell)
        sources.append(source)
        if raw_cell.get("cell_type") != "code":
            continue
        if raw_cell.get("outputs"):
            findings.append(Violation(path, index, "notebook code cell has outputs"))
        if raw_cell.get("execution_count") is not None:
            findings.append(
                Violation(path, index, "notebook code cell has an execution count")
            )
        if NOTEBOOK_SHELL_RE.search(source):
            findings.append(
                Violation(path, index, "notebook runs a nonportable shell command")
            )
    findings.extend(audit_text(path, text))
    reported = {finding.message for finding in findings}
    findings.extend(
        finding
        for finding in audit_text(path, "\n".join(sources))
        if finding.message not in reported
    )
    return findings


def audit_file(path: Path, root: Path) -> list[Violation]:
    relative = path.relative_to(root)
    findings = []
    if LOCAL_MACHINE_RE.search(relative.as_posix()):
        findings.append(
            Violation(path, 1, "file path leaks a known local machine name")
        )
    if (
        GENERATED_PARTS.intersection(relative.parts)
        or path.suffix in GENERATED_SUFFIXES
    ):
        return findings + [
            Violation(path, 1, "generated cache or editor artifact is releasable")
        ]
    if path.name == ".env" or (
        path.name.startswith(".env.") and path.name != ".env.example"
    ):
        return findings + [Violation(path, 1, "environment file must not be released")]
    if path.is_symlink():
        return findings + [
            Violation(path, 1, "symbolic links are not portable release artifacts")
        ]
    if path.suffix.lower() == ".ipynb":
        return findings + audit_notebook(path)
    if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in TEXT_NAMES:
        return findings
    try:
        return findings + audit_text(path, path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError):
        return findings + [
            Violation(path, 1, "text release artifact is not readable UTF-8")
        ]


def audit_tree(root: Path) -> tuple[list[Path], list[Violation]]:
    files = discover_files(root)
    findings = sorted(finding for path in files for finding in audit_file(path, root))
    if not files:
        findings.append(Violation(root, 1, "companion release inventory is empty"))
    return files, findings


def main() -> int:
    root = Path.cwd().resolve()
    files, findings = audit_tree(root)
    if findings:
        print(
            f"[FAIL] companion release hygiene found {len(findings)} issue(s)",
            file=sys.stderr,
        )
        for finding in findings:
            print(
                f"  {finding.path.relative_to(root)}:{finding.line}: {finding.message}",
                file=sys.stderr,
            )
        return 1
    print(f"[OK] companion release hygiene passed ({len(files)} file(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
