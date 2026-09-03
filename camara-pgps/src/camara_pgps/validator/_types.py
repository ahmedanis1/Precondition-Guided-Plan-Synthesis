"""Shared types for plan validation.

Two layers, deliberately kept separate:

* ``structural`` - is the plan a well-formed PGPS plan? (operations exist,
  dependencies satisfied, no cycles, ordering consistent)
* ``payload`` - does each step conform to the CAMARA API contract? (required
  parameters, types, enums, request-body / OpenAPI schema)
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, computed_field


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationLayer(StrEnum):
    STRUCTURAL = "structural"
    PAYLOAD = "payload"


class ValidationCategory(StrEnum):
    # structural
    API_EXISTS = "api_exists"
    OPERATION_EXISTS = "operation_exists"
    ORDERING = "ordering"
    DEPENDENCY = "dependency"
    CYCLE = "cycle"
    PRECONDITION = "precondition"
    # payload
    REQUIRED_PARAMETER = "required_parameter"
    PARAMETER_TYPE = "parameter_type"
    ENUM_VALUE = "enum_value"
    UNKNOWN_PARAMETER = "unknown_parameter"
    PAYLOAD_SCHEMA = "payload_schema"


class ValidationIssue(BaseModel):
    """A single validation finding."""

    layer: ValidationLayer
    category: ValidationCategory
    severity: Severity
    message: str
    step_index: int | None = None
    operation_id: str | None = None
    parameter: str | None = None

    def __str__(self) -> str:
        where = ""
        if self.step_index is not None:
            where = f" [step {self.step_index}"
            if self.operation_id:
                where += f" {self.operation_id}"
            if self.parameter:
                where += f".{self.parameter}"
            where += "]"
        return f"{self.severity.value}: {self.message}{where}"


class ValidationReport(BaseModel):
    """The outcome of validating a plan."""

    issues: list[ValidationIssue] = Field(default_factory=list)
    checked_steps: int = 0

    # -- rollups ------------------------------------------------------------

    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity is Severity.WARNING]

    def for_layer(self, layer: ValidationLayer) -> list[ValidationIssue]:
        return [i for i in self.issues if i.layer is layer]

    def for_step(self, step_index: int) -> list[ValidationIssue]:
        return [i for i in self.issues if i.step_index == step_index]

    def _has_error(self, layer: ValidationLayer) -> bool:
        return any(
            i.severity is Severity.ERROR and i.layer is layer for i in self.issues
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def structural_ok(self) -> bool:
        return not self._has_error(ValidationLayer.STRUCTURAL)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def payload_ok(self) -> bool:
        return not self._has_error(ValidationLayer.PAYLOAD)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ok(self) -> bool:
        return self.structural_ok and self.payload_ok

    def has_category(self, category: ValidationCategory) -> bool:
        return any(i.category is category for i in self.issues)

    def summary(self) -> str:
        error_count = len(self.errors())
        warning_count = len(self.warnings())
        verdict = "valid" if self.ok else "invalid"
        return (
            f"{verdict}: {error_count} error(s), {warning_count} warning(s) "
            f"across {self.checked_steps} step(s) "
            f"(structural_ok={self.structural_ok}, payload_ok={self.payload_ok})"
        )


__all__ = [
    "Severity",
    "ValidationLayer",
    "ValidationCategory",
    "ValidationIssue",
    "ValidationReport",
]
