"""Small schema-definition and instance fixtures for Chapter 27."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeGuard


JSON = Any
LEVELS = ("loose", "lite", "strict", "ultra")
SUPPORTED_TYPES = {"null", "boolean", "object", "array", "number", "integer", "string"}
SUPPORTED_KEYWORDS = {
    "$defs",
    "$ref",
    "type",
    "anyOf",
    "properties",
    "additionalProperties",
    "items",
    "enum",
    "required",
    "description",
    "default",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "pattern",
}

STUDIO_RESPONSE_SCHEMA: dict[str, JSON] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["ready", "not_ready"]},
        "blocking_ids": {"type": "array", "items": {"type": "string"}},
        "evidence_count": {"type": "integer"},
        "summary": {"type": "string"},
    },
    "required": ["verdict", "blocking_ids", "evidence_count", "summary"],
    "additionalProperties": False,
}
STUDIO_RESPONSE: dict[str, JSON] = {
    "verdict": "not_ready",
    "blocking_ids": ["D3"],
    "evidence_count": 11,
    "summary": "Dependency evidence D3 is absent at studio@v7.",
}


@dataclass(frozen=True)
class Diagnostic:
    """Report schema-audit severity and stable diagnostic codes."""

    severity: str
    codes: tuple[str, ...]

    @property
    def acceptable(self) -> bool:
        """The chapter's strict caller policy accepts success only."""

        return self.severity == "success"


@dataclass(frozen=True)
class CanonicalResult:
    """Candidate bytes and diagnostic are one return object."""

    candidate: dict[str, JSON]
    diagnostic: Diagnostic


@dataclass(frozen=True)
class CanonicalizationTrace:
    """Bounded result for callers that require an accepted fixed point."""

    candidate: dict[str, JSON]
    diagnostic: Diagnostic
    attempts: int
    status: str


def _schema_children(schema: Mapping[str, JSON]) -> Iterable[tuple[str, JSON]]:
    for keyword in ("properties", "$defs"):
        children = schema.get(keyword, {})
        if isinstance(children, Mapping):
            for name, child in children.items():
                yield f"{keyword}.{name}", child
    if "items" in schema:
        yield "items", schema["items"]
    alternatives = schema.get("anyOf", [])
    if isinstance(alternatives, Sequence) and not isinstance(
        alternatives, (str, bytes)
    ):
        for index, child in enumerate(alternatives):
            yield f"anyOf.{index}", child
    additional = schema.get("additionalProperties")
    if isinstance(additional, Mapping):
        yield "additionalProperties", additional


def _reference_errors(
    node: Mapping[str, JSON], root_defs: JSON, path: str
) -> list[str]:
    reference = node.get("$ref")
    if reference is None:
        return []
    if not isinstance(reference, str) or not reference.startswith("#"):
        return [f"{path}:external-ref"]
    if reference.startswith("#/$defs/"):
        name = reference.removeprefix("#/$defs/")
        if not isinstance(root_defs, Mapping) or name not in root_defs:
            return [f"{path}:unresolved-ref"]
    return []


def _strict_errors(node: Mapping[str, JSON], path: str) -> list[str]:
    minimum = node.get("minimum")
    maximum = node.get("maximum")
    if _is_number(minimum) and _is_number(maximum) and minimum > maximum:
        return [f"{path}:contradictory-bounds"]
    return []


def _ultra_errors(node: Mapping[str, JSON], path: str) -> list[str]:
    if node.get("type") != "object":
        return []
    required = node.get("required", [])
    properties = node.get("properties", {})
    if not isinstance(required, list) or not all(isinstance(x, str) for x in required):
        return [f"{path}:bad-required"]
    if node.get("additionalProperties") is not False or not isinstance(
        properties, Mapping
    ):
        return []
    return [
        f"{path}:required-without-property:{name}"
        for name in sorted(set(required) - set(properties))
    ]


def _walk_profile(
    node: JSON, path: str, level: str, root_defs: JSON, errors: list[str]
) -> None:
    if not isinstance(node, Mapping):
        errors.append(f"{path}:schema-not-object")
        return
    errors.extend(
        f"{path}:unsupported:{key}" for key in sorted(set(node) - SUPPORTED_KEYWORDS)
    )
    declared_type = node.get("type")
    if declared_type is not None and declared_type not in SUPPORTED_TYPES:
        errors.append(f"{path}:bad-type")
    errors.extend(_reference_errors(node, root_defs, path))
    if level in {"strict", "ultra"}:
        errors.extend(_strict_errors(node, path))
    if level == "ultra":
        errors.extend(_ultra_errors(node, path))
    for child_name, child in _schema_children(node):
        _walk_profile(child, f"{path}.{child_name}", level, root_defs, errors)


def validate_schema_profile(schema: JSON, level: str) -> tuple[str, ...]:
    """Check a transparent teaching subset at one Walle-named level."""

    if level not in LEVELS:
        raise ValueError(f"unknown level: {level}")
    if not isinstance(schema, Mapping):
        return ("schema-not-object",)
    if level == "loose":
        return ()
    errors: list[str] = []
    _walk_profile(schema, "$", level, schema.get("$defs", {}), errors)
    return tuple(errors)


def _is_number(value: JSON) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _scalar_errors(instance: JSON, schema: Mapping[str, JSON], path: str) -> list[str]:
    errors: list[str] = []
    if _is_number(instance):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}:below-minimum")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}:above-maximum")
    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append(f"{path}:short-string")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errors.append(f"{path}:long-string")
    return errors


def _array_errors(
    instance: list[JSON], schema: Mapping[str, JSON], path: str
) -> list[str]:
    errors: list[str] = []
    if "minItems" in schema and len(instance) < schema["minItems"]:
        errors.append(f"{path}:short-array")
    if "maxItems" in schema and len(instance) > schema["maxItems"]:
        errors.append(f"{path}:long-array")
    item_schema = schema.get("items")
    if isinstance(item_schema, Mapping):
        for index, item in enumerate(instance):
            errors.extend(instance_errors(item, item_schema, f"{path}[{index}]"))
    return errors


def _object_errors(
    instance: Mapping[str, JSON], schema: Mapping[str, JSON], path: str
) -> list[str]:
    errors: list[str] = []
    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        properties = {}
    required = schema.get("required", [])
    if isinstance(required, list):
        errors.extend(
            f"{path}:missing:{name}" for name in required if name not in instance
        )
    for name, value in instance.items():
        child_path = f"{path}.{name}"
        if name in properties and isinstance(properties[name], Mapping):
            errors.extend(instance_errors(value, properties[name], child_path))
        elif schema.get("additionalProperties") is False:
            errors.append(f"{path}:additional:{name}")
        elif isinstance(schema.get("additionalProperties"), Mapping):
            errors.extend(
                instance_errors(value, schema["additionalProperties"], child_path)
            )
    return errors


def _type_matches(instance: JSON, declared: JSON) -> bool:
    checks = {
        None: True,
        "null": instance is None,
        "boolean": isinstance(instance, bool),
        "integer": isinstance(instance, int) and not isinstance(instance, bool),
        "number": _is_number(instance),
        "string": isinstance(instance, str),
        "array": isinstance(instance, list),
        "object": isinstance(instance, Mapping),
    }
    return checks.get(declared, False)


def instance_errors(
    instance: JSON, schema: Mapping[str, JSON], path: str = "$"
) -> tuple[str, ...]:
    """Validate only the assertions used by Chapter 27 fixtures."""

    alternatives = schema.get("anyOf")
    if alternatives is not None:
        branches = alternatives if isinstance(alternatives, list) else []
        if not any(
            isinstance(branch, Mapping) and not instance_errors(instance, branch, path)
            for branch in branches
        ):
            return (f"{path}:no-anyOf-branch",)
    errors: list[str] = []
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}:not-in-enum")
    declared = schema.get("type")
    if not _type_matches(instance, declared):
        errors.append(f"{path}:expected-{declared}")
        return tuple(errors)
    errors.extend(_scalar_errors(instance, schema, path))
    if isinstance(instance, list):
        errors.extend(_array_errors(instance, schema, path))
    if isinstance(instance, Mapping):
        errors.extend(_object_errors(instance, schema, path))
    return tuple(errors)


def instance_conforms(instance: JSON, schema: Mapping[str, JSON]) -> bool:
    """Return whether an instance conforms to the schema subset."""
    return not instance_errors(instance, schema)


def _deep_sort(value: JSON) -> JSON:
    if isinstance(value, Mapping):
        return {key: _deep_sort(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_deep_sort(item) for item in value]
    return deepcopy(value)


def _simplify_child(value: JSON) -> JSON:
    if isinstance(value, Mapping):
        return _simplify_for_diagnosis(value)
    return deepcopy(value)


def _simplify_keyword(key: str, value: JSON) -> JSON:
    if key in {"properties", "$defs"} and isinstance(value, Mapping):
        return {name: _simplify_child(child) for name, child in sorted(value.items())}
    if key in {"items", "additionalProperties"} and isinstance(value, Mapping):
        return _simplify_for_diagnosis(value)
    if key == "anyOf" and isinstance(value, list):
        return [_simplify_child(child) for child in value]
    return _deep_sort(value)


def _has_contradictory_bounds(schema: Mapping[str, JSON]) -> bool:
    minimum, maximum = schema.get("minimum"), schema.get("maximum")
    return _is_number(minimum) and _is_number(maximum) and minimum > maximum


def _prune_required(schema: dict[str, JSON]) -> None:
    if (
        schema.get("type") != "object"
        or schema.get("additionalProperties") is not False
    ):
        return
    properties, required = schema.get("properties", {}), schema.get("required", [])
    if isinstance(properties, Mapping) and isinstance(required, list):
        schema["required"] = [name for name in required if name in properties]


def _simplify_for_diagnosis(schema: Mapping[str, JSON]) -> dict[str, JSON]:
    result = {
        key: _simplify_keyword(key, schema[key])
        for key in sorted(schema)
        if key in SUPPORTED_KEYWORDS
    }
    if _has_contradictory_bounds(result):
        return {}
    _prune_required(result)
    return result


def canonicalize_schema(schema: Mapping[str, JSON]) -> CanonicalResult:
    """Return a key-sorted candidate and its non-separable diagnostic."""

    original = deepcopy(schema)
    errors = validate_schema_profile(original, "ultra")
    if errors:
        return CanonicalResult(
            _simplify_for_diagnosis(original), Diagnostic("warning", errors)
        )
    return CanonicalResult(_deep_sort(original), Diagnostic("success", ("accepted",)))


def _representation_bytes(schema: Mapping[str, JSON]) -> bytes:
    """Serialize without sorting so a representation fixed point is observable."""

    return json.dumps(schema, separators=(",", ":"), ensure_ascii=False).encode()


def canonicalize_to_fixed_point(
    schema: Mapping[str, JSON],
    *,
    max_attempts: int = 4,
    step: Callable[[Mapping[str, JSON]], CanonicalResult] = canonicalize_schema,
) -> CanonicalizationTrace:
    """Iterate accepted rewrites to a representation fixed point within a bound."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    current = deepcopy(dict(schema))
    for attempt in range(1, max_attempts + 1):
        result = step(deepcopy(current))
        if not result.diagnostic.acceptable:
            return CanonicalizationTrace(
                deepcopy(result.candidate), result.diagnostic, attempt, "rejected"
            )
        if _representation_bytes(result.candidate) == _representation_bytes(current):
            return CanonicalizationTrace(
                deepcopy(result.candidate), result.diagnostic, attempt, "stable"
            )
        current = deepcopy(result.candidate)
    return CanonicalizationTrace(
        current,
        Diagnostic("success", ("accepted-but-not-stable-within-bound",)),
        max_attempts,
        "max-attempts",
    )


def canonical_bytes(schema: Mapping[str, JSON]) -> bytes:
    """Return the schema's canonical byte representation."""
    return json.dumps(
        _deep_sort(schema), separators=(",", ":"), ensure_ascii=False
    ).encode()


def finite_semantic_relation(
    original: Mapping[str, JSON], candidate: Mapping[str, JSON], domain: Iterable[JSON]
) -> dict[str, JSON]:
    """Classify denotations only over an explicitly finite witness domain."""

    original_set = [value for value in domain if instance_conforms(value, original)]
    candidate_set = [value for value in domain if instance_conforms(value, candidate)]
    left = {json.dumps(value, sort_keys=True) for value in original_set}
    right = {json.dumps(value, sort_keys=True) for value in candidate_set}
    relation = "equal-on-domain"
    if left < right:
        relation = "weakened-on-domain"
    elif right < left:
        relation = "strengthened-on-domain"
    elif left != right:
        relation = "incomparable-on-domain"
    return {
        "relation": relation,
        "original_accepts": original_set,
        "candidate_accepts": candidate_set,
    }
