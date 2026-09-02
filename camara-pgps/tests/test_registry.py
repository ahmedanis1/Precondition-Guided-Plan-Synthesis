"""Tests for the CAMARA API registry (milestone 3)."""

from __future__ import annotations

import pytest

from camara_pgps.camara import (
    CamaraRegistry,
    CapabilityMapper,
    InvalidCapabilityError,
    RegistryDocument,
    SourceType,
    UnknownAPIError,
    UnknownCapabilityError,
    UnknownOperationError,
    get_registry,
)
from camara_pgps.camara.capability_mapper import CAPABILITY_PATTERN
from camara_pgps.config import PROJECT_ROOT
from camara_pgps.models import CamaraAPI, CamaraOperation

REGISTRY_FILE = PROJECT_ROOT / "data" / "registry" / "camara_apis.json"


@pytest.fixture(scope="module")
def registry() -> CamaraRegistry:
    return CamaraRegistry.from_file(REGISTRY_FILE)


def test_registry_file_is_valid() -> None:
    document = RegistryDocument.model_validate_json(REGISTRY_FILE.read_text("utf-8"))
    assert [api.name for api in document.apis] == [
        "device-location",
        "quality-on-demand",
        "network-insights",
    ]


def test_lists_apis_in_file_order(registry: CamaraRegistry) -> None:
    assert registry.api_names() == [
        "device-location",
        "quality-on-demand",
        "network-insights",
    ]
    assert all(isinstance(api, CamaraAPI) for api in registry.list_apis())


def test_operations_of_an_api(registry: CamaraRegistry) -> None:
    assert registry.operation_ids("quality-on-demand") == [
        "createSession",
        "getSession",
        "deleteSession",
    ]
    assert all(
        isinstance(op, CamaraOperation)
        for op in registry.list_operations("quality-on-demand")
    )


def test_required_parameters_of_operation(registry: CamaraRegistry) -> None:
    required = {p.name for p in registry.required_parameters_of("createSession")}
    assert required == {"device", "applicationServer", "qosProfile", "duration"}
    assert {p.name for p in registry.parameters_of("createSession")} >= required


def test_capability_lookup_both_directions(registry: CamaraRegistry) -> None:
    assert registry.capability_of("createSession") == "qos.session.create"
    assert [
        op.operation_id for op in registry.operations_for_capability("qos.session.delete")
    ] == ["deleteSession"]


def test_capabilities_form_a_closed_vocabulary(registry: CamaraRegistry) -> None:
    caps = registry.capabilities()
    assert caps == sorted(caps)
    assert "qos.session.create" in caps
    assert "device-location.verify" in caps
    # Every registered operation is tagged with a well-formed capability.
    for op_id in registry.operation_ids():
        cap = registry.capability_of(op_id)
        assert cap is not None and CAPABILITY_PATTERN.match(cap)


def test_provenance_reports_source(registry: CamaraRegistry) -> None:
    prov = registry.provenance_of("network-insights")
    assert prov.source_type is SourceType.LOCAL_REGISTRY
    assert prov.location == "data/registry/camara_apis.json"
    assert prov.spec_version == "0.1.0"


def test_unknown_api_raises(registry: CamaraRegistry) -> None:
    assert registry.get_api("teleport") is None
    with pytest.raises(UnknownAPIError):
        registry.require_api("teleport")


def test_unknown_operation_raises_the_hallucination_guard(registry: CamaraRegistry) -> None:
    assert registry.has_operation("summonUnicorn") is False
    with pytest.raises(UnknownOperationError):
        registry.require_operation("summonUnicorn")


def test_unknown_capability_raises(registry: CamaraRegistry) -> None:
    assert registry.operations_for_capability("qos.session.teleport") == []
    with pytest.raises(UnknownCapabilityError):
        registry.capability_mapper.require_operations_with("qos.session.teleport")


def test_capability_mapper_validates_format() -> None:
    assert CapabilityMapper.validate(" QOS.Session.Create ") == "qos.session.create"
    with pytest.raises(InvalidCapabilityError):
        CapabilityMapper.validate("not a capability")
    with pytest.raises(InvalidCapabilityError):
        CapabilityMapper.validate("singleword")


def test_get_registry_is_cached() -> None:
    assert get_registry() is get_registry()
