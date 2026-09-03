"""Parameter / payload validation - the API-contract layer.

Answers: are the required parameters present, do the provided values match the
declared JSON-Schema type, are enum values legal, and does the assembled request
body conform to the operation's OpenAPI request schema?

This does NOT look at ordering or dependencies - that is :mod:`structural`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.models import APIParameter, BindingSource, ParameterLocation
from camara_pgps.models.plan import PlanResult, PlanStep
from camara_pgps.validator._types import (
    Severity,
    ValidationCategory,
    ValidationIssue,
    ValidationLayer,
)

_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _issue(
    category: ValidationCategory,
    severity: Severity,
    message: str,
    *,
    step_index: int | None = None,
    operation_id: str | None = None,
    parameter: str | None = None,
) -> ValidationIssue:
    return ValidationIssue(
        layer=ValidationLayer.PAYLOAD,
        category=category,
        severity=severity,
        message=message,
        step_index=step_index,
        operation_id=operation_id,
        parameter=parameter,
    )


def _resolved_argument_names(step: PlanStep) -> set[str]:
    """Parameter names this step will have a value for at execution time."""

    names = set(step.arguments)
    for binding in step.bindings:
        if binding.source is BindingSource.UNRESOLVED:
            continue
        if binding.value is not None or binding.source in (
            BindingSource.PRIOR_STEP,
            BindingSource.DEFAULT,
        ):
            names.add(binding.parameter)
    return names


def _type_ok(value: Any, json_type: str) -> bool:
    expected = _JSON_TYPES.get(json_type)
    if expected is None:
        return True
    if json_type == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, expected)


def _check_value(
    param: APIParameter,
    value: Any,
    step: PlanStep,
    issues: list[ValidationIssue],
) -> None:
    schema = param.schema_ or {}
    json_type = schema.get("type")

    if json_type and not _type_ok(value, json_type):
        scalar = json_type in ("string", "integer", "number", "boolean")
        issues.append(
            _issue(
                ValidationCategory.PARAMETER_TYPE,
                Severity.ERROR if scalar else Severity.WARNING,
                f"expected type {json_type!r} but got {type(value).__name__}",
                step_index=step.index,
                operation_id=step.operation_id,
                parameter=param.name,
            )
        )

    if param.enum:
        allowed = {str(option).lower() for option in param.enum}
        if str(value).lower() not in allowed:
            issues.append(
                _issue(
                    ValidationCategory.ENUM_VALUE,
                    Severity.ERROR,
                    f"value {value!r} is not one of {param.enum}",
                    step_index=step.index,
                    operation_id=step.operation_id,
                    parameter=param.name,
                )
            )


def _check_payload_schema(
    request_schema: dict[str, Any],
    body: dict[str, Any],
    step: PlanStep,
    issues: list[ValidationIssue],
) -> None:
    if not any(key in request_schema for key in ("properties", "required", "type")):
        return
    try:
        import jsonschema
        from jsonschema.validators import validator_for
    except ImportError:  # pragma: no cover - jsonschema is a declared dependency
        issues.append(
            _issue(
                ValidationCategory.PAYLOAD_SCHEMA,
                Severity.WARNING,
                "jsonschema is not installed; skipped OpenAPI payload conformance",
                step_index=step.index,
                operation_id=step.operation_id,
            )
        )
        return

    validator_cls = validator_for(request_schema)
    try:
        validator_cls.check_schema(request_schema)
    except jsonschema.SchemaError:
        return  # a malformed reference schema is not the plan's fault

    for error in validator_cls(request_schema).iter_errors(body):
        pointer = "/".join(str(part) for part in error.absolute_path) or "<body>"
        issues.append(
            _issue(
                ValidationCategory.PAYLOAD_SCHEMA,
                Severity.ERROR,
                f"request body does not conform at {pointer}: {error.message}",
                step_index=step.index,
                operation_id=step.operation_id,
            )
        )


def check_parameters(
    plan: PlanResult,
    registry: CamaraRegistry,
    *,
    arguments: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    extra_arguments = arguments or {}

    for step in plan.steps:
        operation = registry.get_operation(step.operation_id)
        if operation is None:
            continue  # structural layer already reported this

        provided: dict[str, Any] = dict(step.arguments)
        provided.update(extra_arguments.get(step.operation_id, {}))
        will_have = _resolved_argument_names(step) | set(provided)
        by_name = {param.name: param for param in operation.parameters}

        # required parameters present?
        for param in operation.parameters:
            if param.required and param.name not in will_have:
                issues.append(
                    _issue(
                        ValidationCategory.REQUIRED_PARAMETER,
                        Severity.ERROR,
                        f"required parameter {param.name!r} has no value or binding",
                        step_index=step.index,
                        operation_id=step.operation_id,
                        parameter=param.name,
                    )
                )

        # value-level checks for the values we actually have
        for name, value in provided.items():
            declared = by_name.get(name)
            if declared is None:
                issues.append(
                    _issue(
                        ValidationCategory.UNKNOWN_PARAMETER,
                        Severity.WARNING,
                        f"{name!r} is not a parameter of {step.operation_id!r}",
                        step_index=step.index,
                        operation_id=step.operation_id,
                        parameter=name,
                    )
                )
                continue
            _check_value(declared, value, step, issues)

        # request-body / OpenAPI schema conformance
        if operation.request_schema:
            body = {
                name: provided[name]
                for name, param in by_name.items()
                if param.location is ParameterLocation.BODY and name in provided
            }
            if body:
                _check_payload_schema(operation.request_schema, body, step, issues)

    return issues


__all__ = ["check_parameters"]
