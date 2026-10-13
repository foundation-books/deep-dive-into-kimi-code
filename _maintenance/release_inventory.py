"""Validate the reviewed standalone release inventory and documentation links."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re
from urllib.parse import unquote, urlsplit

from .release_hygiene import SKIP_DIRS, discover_files


INVENTORY_NAME = "release_inventory.json"
MANIFEST_NAME = "release-manifest.json"
MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
HTML_LINK_RE = re.compile(r"(?:src|href)=[\"\']([^\"\']+)[\"\']")


class ReleaseError(ValueError):
    """A companion tree violates the public release contract."""


@dataclass(frozen=True)
class Inventory:
    """Files reviewed for inclusion and author-only files omitted from exports."""

    included: frozenset[str]
    excluded: frozenset[str]


def _portable_name(name: object) -> bool:
    if not isinstance(name, str) or not name or re.search(r"[\x00-\x1f\\:]", name):
        return False
    relative = PurePosixPath(name)
    return (
        not relative.is_absolute()
        and relative.as_posix() == name
        and name != "."
        and ".." not in relative.parts
        and SKIP_DIRS.isdisjoint(relative.parts)
        and name != MANIFEST_NAME
    )


def load_inventory(root: Path) -> Inventory:
    """Read exact, portable included paths and documented exclusions."""

    try:
        payload = json.loads((root / INVENTORY_NAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReleaseError(f"cannot read {INVENTORY_NAME}: {error}") from error
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "included", "excluded"}
        or payload["schema_version"] != 1
    ):
        raise ReleaseError(
            "inventory requires schema_version 1, included, and excluded"
        )
    included, excluded = payload["included"], payload["excluded"]
    if not isinstance(included, list) or not included:
        raise ReleaseError("inventory included must be a nonempty list")
    if not isinstance(excluded, dict) or not all(
        isinstance(reason, str) and reason.strip() for reason in excluded.values()
    ):
        raise ReleaseError("inventory excluded must map paths to nonempty reasons")
    if not all(_portable_name(name) for name in [*included, *excluded]):
        raise ReleaseError("inventory paths must be portable relative file names")
    if len(included) != len(set(included)):
        raise ReleaseError("duplicate included file in inventory")
    selected, omitted = frozenset(included), frozenset(excluded)
    if selected & omitted:
        raise ReleaseError("included and excluded files overlap")
    if INVENTORY_NAME not in selected:
        raise ReleaseError(f"inventory must include {INVENTORY_NAME}")
    return Inventory(selected, omitted)


def _link_failure(
    root: Path, name: str, target: str, included: frozenset[str]
) -> str | None:
    target = target.strip().strip("<>")
    parsed = urlsplit(target)
    if parsed.scheme in {"https", "http", "mailto"}:
        return None
    if parsed.scheme or parsed.netloc or target.startswith("/"):
        return f"{name}: unsupported documentation link {target!r}"
    if not parsed.path:
        return None
    resolved = (root / name).parent / unquote(parsed.path)
    try:
        relative = resolved.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return f"{name}: documentation link escapes code tree: {target!r}"
    if relative not in included:
        return f"{name}: documentation link target is not exported: {target!r}"
    return None


def documentation_failures(root: Path, included: frozenset[str]) -> list[str]:
    """Check Markdown links and HTML image sources in all released documentation."""

    failures: list[str] = []
    for name in sorted(included):
        if not name.endswith(".md"):
            continue
        text = (root / name).read_text(encoding="utf-8")
        targets = [match.group(1) for match in MARKDOWN_LINK_RE.finditer(text)]
        targets.extend(match.group(1) for match in HTML_LINK_RE.finditer(text))
        failures.extend(
            failure
            for target in targets
            if (failure := _link_failure(root, name, target, included)) is not None
        )
    return failures


def validate_inventory(root: Path) -> Inventory:
    """Reject unclassified files, missing files, symlinks, and broken local links."""

    inventory = load_inventory(root)
    discovered = discover_files(root)
    found = {item.relative_to(root).as_posix() for item in discovered}
    allowed = inventory.included | inventory.excluded | {MANIFEST_NAME}
    failures = [f"unclassified file: {name}" for name in sorted(found - allowed)]
    failures.extend(
        f"missing included file: {name}" for name in sorted(inventory.included - found)
    )
    failures.extend(
        f"symlink is not portable: {item.relative_to(root).as_posix()}"
        for item in discovered
        if item.is_symlink()
    )
    if not failures:
        failures.extend(documentation_failures(root, inventory.included))
    if failures:
        raise ReleaseError("\n".join(failures))
    return inventory
