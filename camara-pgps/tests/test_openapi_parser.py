"""Tests for the CAMARA OpenAPI parser (milestone 4)."""

from __future__ import annotations

import pytest

from camara_pgps.camara import (
    CamaraRegistry,
    OpenAPIParseError,
    OpenAPIParser,
    RegistryAPI,
    build_registry,
    parse_openapi,
)
from camara_pgps.models import CamaraAPI, HTTPMethod, ParameterLocation

FIXTURE = "tests/fixtures/quality_on_demand.openapi.yaml"
SOURCE_URL = "https://raw.example.com/camara/quality-on-demand.yaml"


@pytest.fixture(scope="module")
def api() -> RegistryAPI:
    return OpenAPIParser().parse_file(FIXTURE, source_url=SOURCE_URL)


def test_api_level_fields(api: RegistryAPI) -> None:
    assert isinstance(api, CamaraAPI)
    assert api.name == "quality-on-demand"  # derived from the server URL path
    assert api.version == "0.11.0"
    assert api.description and "Quality-on-Demand" in api.description
    assert api.operations and {op.operation_id for op in api.operations} == {
        "createSession",
        "getSession",
        "deleteSession",
    }


def test_methods_and_paths(api: RegistryAPI) -> None:
    by_id = {op.operation_id: op for op in api.operations}
    assert by_id["createSession"].method is HTTPMethod.POST
    assert by_id["createSession"].path == "/sessions"
    assert by_id["getSession"].method is HTTPMethod.GET
    assert by_id["deleteSession"].method is HTTPMethod.DELETE
    assert by_id["getSession"].path == "/sessions/{sessionId}"


def test_request_body_params_required_fields_and_enums(api: RegistryAPI) -> None:
    create = next(op for op in api.operations if op.operation_id == "createSession")
    body = {p.name: p for p in create.parameters if p.location is ParameterLocation.BODY}
    assert set(body) == {"device", "applicationServer", "qosProfile", "duration"}
    assert {name for name, p in body.items() if p.required} == {
        "device",
        "qosProfile",
        "duration",
    }
    assert body["qosProfile"].enum == ["QOS_E", "QOS_S", "QOS_M", "QOS_L"]
    assert create.request_schema is not None
    assert "properties" in create.request_schema


def test_ref_resolution_inlines_nested_schemas(api: RegistryAPI) -> None:
    create = next(op for op in api.operations if op.operation_id == "createSession")
    device_schema = create.request_schema["properties"]["device"]
    # #/components/schemas/Device was dereferenced, not left as a $ref.
    assert "$ref" not in device_schema
    assert "phoneNumber" in device_schema["properties"]


def test_response_and_error_extraction(api: RegistryAPI) -> None:
    create = next(op for op in api.operations if op.operation_id == "createSession")
    assert create.response_schema is not None  # the 201 body schema
    assert {r.status_code for r in create.errors} == {"400", "401", "429"}
    assert all(r.is_error for r in create.errors)
    # the $ref'd 400 response was resolved to its ErrorInfo schema
    generic_400 = next(r for r in create.responses if r.status_code == "400")
    assert generic_400.schema_ is not None


def test_security_scopes(api: RegistryAPI) -> None:
    by_id = {op.operation_id: op for op in api.operations}
    assert by_id["createSession"].security_scopes == ["quality-on-demand:sessions:create"]
    # getSession inherits the document-level requirement
    assert by_id["getSession"].security_scopes == ["quality-on-demand:sessions:read"]


def test_path_level_parameter_is_merged(api: RegistryAPI) -> None:
    get_session = next(op for op in api.operations if op.operation_id == "getSession")
    session_id = get_session.get_parameter("sessionId")
    assert session_id is not None
    assert session_id.location is ParameterLocation.PATH
    assert session_id.required is True


def test_provenance_and_source_ref_are_preserved(api: RegistryAPI) -> None:
    assert api.provenance.source_type == "openapi"
    assert api.provenance.source_url == SOURCE_URL
    assert api.provenance.spec_version == "0.11.0"
    assert api.spec_url == SOURCE_URL
    assert api.source_file is not None

    create = next(op for op in api.operations if op.operation_id == "createSession")
    assert create.source is not None
    assert create.source.api_name == "quality-on-demand"
    assert create.source.version == "0.11.0"
    assert create.source.operation_id == "createSession"
    assert create.source.source_url == SOURCE_URL


def test_examples_are_collected(api: RegistryAPI) -> None:
    create = next(op for op in api.operations if op.operation_id == "createSession")
    assert create.examples.get("request") == {"qosProfile": "QOS_E", "duration": 3600}
    assert "201" in create.examples.get("responses", {})


def test_parse_openapi_accepts_a_mapping() -> None:
    doc = {
        "openapi": "3.0.0",
        "info": {"title": "Tiny API", "version": "1.0.0"},
        "paths": {
            "/ping": {
                "get": {"operationId": "ping", "responses": {"200": {"description": "ok"}}}
            }
        },
    }
    parsed = parse_openapi(doc, name="tiny")
    assert parsed.name == "tiny"
    assert parsed.operations[0].operation_id == "ping"


def test_missing_operation_id_warns_and_synthesises() -> None:
    parser = OpenAPIParser()
    doc = {
        "openapi": "3.0.0",
        "info": {"title": "No Ids", "version": "1.0.0"},
        "paths": {"/things/{id}": {"get": {"responses": {"200": {"description": "ok"}}}}},
    }
    parsed = parser.parse(doc)
    assert parsed.operations[0].operation_id == "get_things_id"
    assert any("operationId" in w for w in parser.warnings)


def test_strict_mode_raises_on_warning() -> None:
    doc = {
        "openapi": "3.0.0",
        "info": {"title": "No Ids", "version": "1.0.0"},
        "paths": {"/x": {"get": {"responses": {"200": {"description": "ok"}}}}},
    }
    with pytest.raises(OpenAPIParseError):
        OpenAPIParser(strict=True).parse(doc)


def test_swagger_2_is_rejected() -> None:
    with pytest.raises(OpenAPIParseError):
        OpenAPIParser().parse({"swagger": "2.0", "info": {"title": "old", "version": "1"}})


def test_build_registry_feeds_the_registry() -> None:
    registry = build_registry([FIXTURE], version="0.1.0")
    assert isinstance(registry, CamaraRegistry)
    assert registry.api_names() == ["quality-on-demand"]
    op = registry.require_operation("createSession")
    assert op.method is HTTPMethod.POST
    # no capability is inferred from OpenAPI
    assert registry.capability_of("createSession") is None
