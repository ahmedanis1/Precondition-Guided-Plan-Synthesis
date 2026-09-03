"""Plan validator: runs the structural and payload layers and updates the plan.

    PGPS      -> structural plan validity   (structural.check_structure)
    Validator -> API / payload validity     (parameters.check_parameters)

``validate`` returns a :class:`ValidationReport` with both layers separated.
``validate_and_update`` also returns a copy of the plan with its ``status`` and
diagnostic fields updated.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import lru_cache
from typing import Any

from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.config import Settings, get_settings
from camara_pgps.models.plan import PlanResult, PlanStatus
from camara_pgps.pgps.conditions import ConditionModel
from camara_pgps.validator._types import ValidationCategory, ValidationReport
from camara_pgps.validator.parameters import check_parameters
from camara_pgps.validator.structural import check_structure


class PlanValidator:
    """Validates a :class:`PlanResult` at both layers."""

    def __init__(
        self,
        *,
        registry: CamaraRegistry | None = None,
        condition_model: ConditionModel | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or CamaraRegistry.from_settings(self.settings)
        self.model = condition_model or ConditionModel.from_registry(self.registry)

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> PlanValidator:
        return cls(settings=settings or get_settings())

    # -- public -----------------------------------------------------------

    def validate(
        self,
        plan: PlanResult,
        *,
        known_facts: Iterable[str] = (),
        parameters: Any | None = None,
    ) -> ValidationReport:
        arguments = self._merge_arguments(parameters)
        issues = check_structure(
            plan,
            self.registry,
            condition_model=self.model,
            known_facts=known_facts,
        )
        issues += check_parameters(plan, self.registry, arguments=arguments)
        return ValidationReport(issues=issues, checked_steps=len(plan.steps))

    def validate_and_update(
        self,
        plan: PlanResult,
        *,
        known_facts: Iterable[str] = (),
        parameters: Any | None = None,
    ) -> tuple[PlanResult, ValidationReport]:
        report = self.validate(plan, known_facts=known_facts, parameters=parameters)
        return self._apply(plan, report), report

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _merge_arguments(parameters: Any | None) -> dict[str, dict[str, Any]]:
        if parameters is None:
            return {}
        # a ParameterExtraction (milestone 9) or anything with .operations[*].arguments
        operations = getattr(parameters, "operations", None)
        if operations is not None:
            return {
                item.operation_id: dict(item.arguments)
                for item in operations
                if getattr(item, "arguments", None)
            }
        if isinstance(parameters, Mapping):
            return {str(k): dict(v) for k, v in parameters.items()}
        raise TypeError(
            "parameters must be a ParameterExtraction or a "
            "Mapping[operation_id, Mapping[param, value]]"
        )

    @staticmethod
    def _apply(plan: PlanResult, report: ValidationReport) -> PlanResult:
        errors = report.errors()
        warnings = [str(issue) for issue in report.warnings()]

        if not errors:
            status = PlanStatus.VALID
        elif report.has_category(ValidationCategory.CYCLE):
            status = PlanStatus.INVALID
        elif not report.structural_ok and any(
            issue.category
            in (ValidationCategory.PRECONDITION, ValidationCategory.DEPENDENCY)
            for issue in errors
        ):
            status = PlanStatus.INFEASIBLE
        else:
            status = PlanStatus.INVALID

        unmet: list[str] = []
        for issue in errors:
            if issue.category is ValidationCategory.PRECONDITION:
                # message form: "unmet preconditions ['a', 'b']: ..."
                start = issue.message.find("[")
                end = issue.message.find("]")
                if 0 <= start < end:
                    unmet.extend(
                        token.strip().strip("'\"")
                        for token in issue.message[start + 1 : end].split(",")
                        if token.strip()
                    )

        return plan.model_copy(
            update={
                "status": status,
                "validation_errors": [str(issue) for issue in errors],
                "unmet_preconditions": sorted(set(unmet)),
                "warnings": [*plan.warnings, *warnings],
                "metadata": {
                    **plan.metadata,
                    "validation": report.summary(),
                    "structural_ok": report.structural_ok,
                    "payload_ok": report.payload_ok,
                },
            }
        )


@lru_cache(maxsize=1)
def get_validator() -> PlanValidator:
    """Return the process-wide plan validator from configured settings."""

    return PlanValidator.from_settings()


def validate_plan(
    plan: PlanResult,
    *,
    registry: CamaraRegistry | None = None,
    known_facts: Iterable[str] = (),
    parameters: Any | None = None,
) -> ValidationReport:
    """One-shot convenience wrapper."""

    validator = (
        PlanValidator(registry=registry) if registry is not None else get_validator()
    )
    return validator.validate(plan, known_facts=known_facts, parameters=parameters)


__all__ = ["PlanValidator", "get_validator", "validate_plan"]
