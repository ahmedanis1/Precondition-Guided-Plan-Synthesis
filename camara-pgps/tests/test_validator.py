"""Tests for plan validation (milestone 12)."""

from __future__ import annotations

import pytest

from camara_pgps.camara import build_registry
from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.config import PROJECT_ROOT
from camara_pgps.models import HTTPMethod
from camara_pgps.models.plan import (
    BindingSource,
    ParameterBinding,
    PlanResult,
    PlanStatus,
    PlanStep,
)
from camara_pgps.pgps import PGPSPlanner
from camara_pgps.validator import (
    PlanValidator,
    ValidationCategory,
    ValidationLayer,
    ValidationReport,
)

REGISTRY_FILE = PROJECT_ROOT / "data" / "registry" / "camara_apis.json"
OPENAPI_FIXTURE = "tests/fixtures/quality_on_demand.openapi.yaml"

QOS_FACTS = {"auth.token", "device.identifier", "qos.profile"}


@pytest.fixture(scope="module")
def registry() -> CamaraRegistry:
    return CamaraRegistry.from_file(REGISTRY_FILE)


@pytest.fixture(scope="module")
def validator(registry: CamaraRegistry) -> PlanValidator:
    return PlanValidator(registry=registry)


@pytest.fixture(scope="module")
def valid_plan(registry: CamaraRegistry) -> PlanResult:
    return PGPSPlanner(registry=registry).plan_for(
        goals=["qos.session.deleted"],
        initial_facts=sorted(QOS_FACTS),
        candidate_operations=["createSession", "deleteSession"],
    )


def _step(
    index: int,
    operation_id: str,
    method: HTTPMethod = HTTPMethod.POST,
    path: str = "/sessions",
    **kwargs: object,
) -> PlanStep:
    return PlanStep(
        index=index, operation_id=operation_id, method=method, path=path, **kwargs
    )


# --------------------------------------------------------------------------- #
# the two layers are reported separately
# --------------------------------------------------------------------------- #

def test_structural_valid_but_payload_incomplete(
    validator: PlanValidator, valid_plan: PlanResult
) -> None:
    report = validator.validate(valid_plan, known_facts=QOS_FACTS)
    assert isinstance(report, ValidationReport)
    assert report.structural_ok is True  # PGPS produced a well-formed plan
    assert report.payload_ok is False  # ...but no parameter values yet
    assert report.ok is False
    assert all(
        i.layer is ValidationLayer.PAYLOAD for i in report.errors()
    )
    assert report.has_category(ValidationCategory.REQUIRED_PARAMETER)


def test_fully_specified_plan_is_valid(
    validator: PlanValidator, valid_plan: PlanResult
) -> None:
    arguments = {
        "createSession": {
            "device": {"phoneNumber": "+12025550123"},
            "applicationServer": {"ipv4Address": "10.0.0.1"},
            "qosProfile": "QOS_L",
            "duration": 3600,
        },
        "deleteSession": {"sessionId": "3fa85f64-5717-4562-b3fc-2c963f66afa6"},
    }
    updated, report = validator.validate_and_update(
        valid_plan, known_facts=QOS_FACTS, parameters=arguments
    )
    assert report.ok is True
    assert report.structural_ok and report.payload_ok
    assert updated.status is PlanStatus.VALID
    assert updated.metadata["payload_ok"] is True


# --------------------------------------------------------------------------- #
# structural layer
# --------------------------------------------------------------------------- #

def test_unknown_operation_is_rejected(validator: PlanValidator) -> None:
    plan = PlanResult(steps=[_step(0, "teleportDevice", HTTPMethod.GET, "/x")])
    report = validator.validate(plan)
    assert report.structural_ok is False
    assert report.has_category(ValidationCategory.OPERATION_EXISTS)


def test_unmet_precondition_is_infeasible(validator: PlanValidator) -> None:
    plan = PlanResult(
        steps=[_step(0, "deleteSession", HTTPMethod.DELETE, "/sessions/{sessionId}")]
    )
    updated, report = validator.validate_and_update(plan, known_facts=set())
    assert report.has_category(ValidationCategory.PRECONDITION)
    assert updated.status is PlanStatus.INFEASIBLE
    assert "qos.session.id" in updated.unmet_preconditions


def test_depends_on_cycle_is_detected(validator: PlanValidator) -> None:
    plan = PlanResult(
        steps=[
            _step(0, "createSession", depends_on=[1]),
            _step(1, "getSession", HTTPMethod.GET, "/sessions/{sessionId}", depends_on=[0]),
        ]
    )
    updated, report = validator.validate_and_update(plan, known_facts=QOS_FACTS)
    assert report.has_category(ValidationCategory.CYCLE)
    assert updated.status is PlanStatus.INVALID


def test_bad_depends_on_and_indices(validator: PlanValidator) -> None:
    plan = PlanResult(
        steps=[
            _step(0, "createSession", depends_on=[7]),
            _step(2, "getSession", HTTPMethod.GET, "/sessions/{sessionId}"),
        ]
    )
    report = validator.validate(plan, known_facts=QOS_FACTS)
    categories = {i.category for i in report.errors()}
    assert ValidationCategory.DEPENDENCY in categories
    assert ValidationCategory.ORDERING in categories


def test_stale_method_path_is_a_warning(validator: PlanValidator) -> None:
    plan = PlanResult(steps=[_step(0, "createSession", HTTPMethod.PUT, "/old/path")])
    report = validator.validate(plan, known_facts=QOS_FACTS)
    assert report.structural_ok is True  # only a warning
    assert any(
        i.category is ValidationCategory.OPERATION_EXISTS and i.severity.value == "warning"
        for i in report.issues
    )


def test_missing_depends_on_is_a_warning(validator: PlanValidator) -> None:
    plan = PlanResult(
        steps=[
            _step(0, "createSession"),
            _step(
                1, "deleteSession", HTTPMethod.DELETE, "/sessions/{sessionId}",
                depends_on=[],  # should reference step 0 (produces qos.session.id)
            ),
        ]
    )
    report = validator.validate(plan, known_facts=QOS_FACTS)
    assert report.structural_ok is True
    assert any(
        i.category is ValidationCategory.DEPENDENCY and i.severity.value == "warning"
        for i in report.issues
    )


# --------------------------------------------------------------------------- #
# payload layer
# --------------------------------------------------------------------------- #

def test_enum_and_type_violations(validator: PlanValidator) -> None:
    plan = PlanResult(steps=[_step(0, "createSession")])
    report = validator.validate(
        plan,
        known_facts=QOS_FACTS,
        parameters={
            "createSession": {
                "device": {"phoneNumber": "+1"},
                "applicationServer": {},
                "qosProfile": "QOS_ULTRA",  # not in enum
                "duration": "soon",  # not an integer
            }
        },
    )
    categories = {i.category for i in report.errors()}
    assert ValidationCategory.ENUM_VALUE in categories
    assert ValidationCategory.PARAMETER_TYPE in categories


def test_object_shaped_scalar_is_only_a_warning(validator: PlanValidator) -> None:
    plan = PlanResult(steps=[_step(0, "createSession")])
    report = validator.validate(
        plan,
        known_facts=QOS_FACTS,
        parameters={
            "createSession": {
                "device": "+12025550123",  # schema says object -> warning, not error
                "applicationServer": {},
                "qosProfile": "QOS_L",
                "duration": 3600,
            }
        },
    )
    type_issues = [
        i for i in report.issues if i.category is ValidationCategory.PARAMETER_TYPE
    ]
    assert type_issues and all(i.severity.value == "warning" for i in type_issues)
    assert report.payload_ok is True


def test_unknown_parameter_is_a_warning(validator: PlanValidator) -> None:
    plan = PlanResult(
        steps=[
            _step(
                0, "getSession", HTTPMethod.GET, "/sessions/{sessionId}",
                arguments={"sessionId": "x", "bogus": 1},
            )
        ]
    )
    report = validator.validate(plan, known_facts={"auth.token", "qos.session.id"})
    assert report.has_category(ValidationCategory.UNKNOWN_PARAMETER)
    assert report.payload_ok is True


def test_prior_step_binding_satisfies_required_param(validator: PlanValidator) -> None:
    plan = PlanResult(
        steps=[
            PlanStep(
                index=0,
                operation_id="getSession",
                method=HTTPMethod.GET,
                path="/sessions/{sessionId}",
                bindings=[
                    ParameterBinding(
                        parameter="sessionId",
                        source=BindingSource.PRIOR_STEP,
                        from_step=0,
                    )
                ],
            )
        ]
    )
    report = validator.validate(plan, known_facts={"auth.token", "qos.session.id"})
    assert not report.has_category(ValidationCategory.REQUIRED_PARAMETER)


# --------------------------------------------------------------------------- #
# OpenAPI request-body conformance
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def openapi_registry() -> CamaraRegistry:
    return build_registry([OPENAPI_FIXTURE], version="0.1.0")


def test_payload_schema_conformance(openapi_registry: CamaraRegistry) -> None:
    validator = PlanValidator(registry=openapi_registry)
    op = openapi_registry.require_operation("createSession")
    assert op.request_schema is not None  # the parsed CreateSession schema

    bad = PlanResult(steps=[_step(0, "createSession", op.method, op.path)])
    report = validator.validate(
        bad,
        known_facts=QOS_FACTS,
        parameters={"createSession": {"qosProfile": "QOS_L", "duration": 3600}},
    )
    assert report.has_category(ValidationCategory.PAYLOAD_SCHEMA)
    assert any("device" in i.message for i in report.errors())

    good = PlanResult(steps=[_step(0, "createSession", op.method, op.path)])
    report = validator.validate(
        good,
        known_facts=QOS_FACTS,
        parameters={
            "createSession": {
                "device": {"phoneNumber": "+12025550123"},
                "qosProfile": "QOS_L",
                "duration": 3600,
            }
        },
    )
    assert not report.has_category(ValidationCategory.PAYLOAD_SCHEMA)
