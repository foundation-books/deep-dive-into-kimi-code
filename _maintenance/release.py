"""Check, export, or verify a code-only local release without manuscript tooling."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import TypedDict

from .release_hygiene import audit_file, discover_files
from .release_inventory import MANIFEST_NAME, ReleaseError, validate_inventory


class ReleaseManifest(TypedDict):
    """Git provenance and SHA-256 hashes of the actual exported file bytes."""

    schema_version: int
    source_commit: str | None
    files: dict[str, str]


def _hashes(root: Path, included: frozenset[str]) -> dict[str, str]:
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in sorted(included)
    }


def _source_commit(root: Path) -> str | None:
    if shutil.which("git") is None:
        return None
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _load_manifest(root: Path) -> ReleaseManifest:
    payload = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "source_commit", "files"}
        or payload["schema_version"] != 1
        or not isinstance(payload["files"], dict)
    ):
        raise ReleaseError("release manifest has an invalid schema")
    commit = payload["source_commit"]
    if commit is not None and (
        not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40,64}", commit) is None
    ):
        raise ReleaseError("release manifest source_commit must be a Git hash or null")
    hashes = payload["files"]
    if not all(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
        for value in hashes.values()
    ):
        raise ReleaseError("release manifest contains an invalid SHA-256 hash")
    return ReleaseManifest(schema_version=1, source_commit=commit, files=hashes)


def verify_release(root: Path) -> ReleaseManifest:
    """Require an exact inventory and matching hashes for every exported file."""

    inventory = validate_inventory(root)
    leaked = sorted(
        item.relative_to(root).as_posix()
        for item in discover_files(root)
        if item.relative_to(root).as_posix() in inventory.excluded
    )
    if leaked:
        raise ReleaseError("excluded file present in release: " + ", ".join(leaked))
    manifest = _load_manifest(root)
    if set(manifest["files"]) != inventory.included:
        raise ReleaseError("release manifest file set differs from included inventory")
    actual = _hashes(root, inventory.included)
    mismatches = [
        name for name, digest in actual.items() if manifest["files"][name] != digest
    ]
    if mismatches:
        raise ReleaseError("release hash mismatch: " + ", ".join(mismatches))
    return manifest


def _check_hygiene(root: Path, included: frozenset[str]) -> None:
    findings = [
        finding
        for name in sorted(included)
        for finding in audit_file(root / name, root)
    ]
    if findings:
        raise ReleaseError(
            "\n".join(
                f"{item.path.relative_to(root)}:{item.line}: {item.message}"
                for item in findings
            )
        )


def build_candidate(root: Path, output: Path) -> ReleaseManifest:
    """Copy the closed inventory to a new directory and reject source drift."""

    root = root.resolve()
    inventory = validate_inventory(root)
    _check_hygiene(root, inventory.included)
    if (root / MANIFEST_NAME).exists():
        verify_release(root)
    if output.exists() or output.is_symlink():
        raise ReleaseError(f"destination already exists: {output}")
    destination = output.resolve()
    if destination == root or root in destination.parents:
        raise ReleaseError("destination must be outside the companion source tree")
    if not destination.parent.is_dir():
        raise ReleaseError("destination parent directory must already exist")
    manifest = ReleaseManifest(
        schema_version=1,
        source_commit=_source_commit(root),
        files=_hashes(root, inventory.included),
    )
    with tempfile.TemporaryDirectory(
        prefix="kimi-companion-export.", dir=destination.parent
    ) as raw:
        staging = Path(raw)
        for name in sorted(inventory.included):
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, target)
        if _hashes(staging, inventory.included) != manifest["files"]:
            raise ReleaseError("source changed while copying release files")
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        verify_release(staging)
        if _hashes(root, inventory.included) != manifest["files"]:
            raise ReleaseError("source changed during export validation")
        # copytree exclusively creates the destination; it cannot overwrite an existing tree.
        shutil.copytree(staging, destination)
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Validate the local inventory, create a candidate, or verify its hashes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--verify", action="store_true")
    action.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        if args.output is not None:
            manifest = build_candidate(root, args.output)
            print(f"[OK] exported {len(manifest['files'])} files to {args.output}")
        elif args.verify:
            manifest = verify_release(root)
            print(f"[OK] release hashes verified ({len(manifest['files'])} files)")
        else:
            inventory = validate_inventory(root)
            if (root / MANIFEST_NAME).exists():
                verify_release(root)
            print(
                f"[OK] release inventory passed ({len(inventory.included)} included files)"
            )
    except (ReleaseError, OSError, UnicodeError, json.JSONDecodeError) as error:
        print(f"[FAIL] companion release: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
