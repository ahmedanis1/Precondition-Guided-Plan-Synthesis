"""Tests for the CAMARA OpenAPI parser (milestone 4).

Rewritten after ``openapi_parser.py`` was accidentally deleted and rebuilt
from scratch in a later session (see docs/pgps-conditions-plan.md) - this
file is a fresh reconstruction against the real fixture and the real
``PlanValidator`` contract, not a byte-for-byte restoration of the original.
"""

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
    assert api.name == "quality-on-demand"
    assert api.version == "0.11.0"
    assert {op.operation_id for op in api.operations} == {
        "createSession",
        "getSession",
        "deleteSession",
    }


def test_methods_and_paths(api: RegistryAPI) -> None:
    by_id = {op.operation_id: op for op in api.operations}
    assert by_id["createSession"].method is HTTPMethod.POST
    assert by_id["createSession"].path == "/sessions"
    assert by_id["getSession"].method is HTTPMethod.GET
    assert by_id["getSession"].path == "/sessions/{sessionId}"
    assert by_id["deleteSession"].method is HTTPMethod.DELETE
    assert by_id["deleteSession"].path == "/sessions/{sessionId}"


def test_request_body_params_required_fields_and_enums(api: RegistryAPI) -> None:
    by_id = {op.operation_id: op for op in api.operations}
    create = by_id["createSession"]
    by_name = {p.name: p for p in create.parameters}

    assert by_name["device"].location is ParameterLocation.BODY
    assert by_name["device"].required is True
    assert by_name["qosProfile"].required is True
    assert by_name["qosProfile"].enum == ["QOS_E", "QOS_S", "QOS_M", "QOS_L"]
    assert by_name["duration"].required is True
    assert by_name["applicationServer"].required is False


def test_ref_resolution_inlines_nested_schemas(api: RegistryAPI) -> None:
    by_id = {op.operation_id: op for op in api.operations}
    device_schema = by_id["createSession"].request_schema["properties"]["device"]
    assert "$ref" not in device_schema
    assert "phoneNumber" in device_schema["properties"]


def test_response_and_error_extraction(api: RegistryAPI) -> None:
    by_id = {op.operation_id: op for op in api.operations}
    create = by_id["createSession"]
    assert create.response_schema is not None
    status_codes = {r.status_code for r in create.responses + create.errors}
    assert {"201", "400", "401", "429"} <= status_codes
    assert all(r.is_error for r in create.errors)
    assert all(not r.is_error for r in create.responses)


def test_security_scopes(api: RegistryAPI) -> None:
    by_id = {op.operation_id: op for op in api.operations}
    # operation-level security overrides the document-level default
    assert "quality-on-demand:sessions:create" in by_id["createSession"].security_scopes
    # getSession/deleteSession declare no operation-level security -> falls
    # back to the document-level default
    assert "quality-on-demand:sessions:read" in by_id["getSession"].security_scopes


def test_path_level_parameter_is_merged(api: RegistryAPI) -> None:
    by_id = {op.operation_id: op for op in api.operations}
    get_session = by_id["getSession"]
    session_id = get_session.get_parameter("sessionId")
    assert session_id is not None
    assert session_id.location is ParameterLocation.PATH
    assert session_id.required is True
    # the same shared path-item parameter is merged into both operations
    delete_session = by_id["deleteSession"]
    assert delete_session.get_parameter("sessionId") is not None


def test_provenance_and_source_ref_are_preserved(api: RegistryAPI) -> None:
    assert api.provenance.source_url == SOURCE_URL
    by_id = {op.operation_id: op for op in api.operations}
    create = by_id["createSession"]
    assert create.source is not None
    assert create.source.api_name == "quality-on-demand"
    assert create.source.version == "0.11.0"
    assert create.source.operation_id == "createSession"
    assert create.source.source_url == SOURCE_URL


def test_examples_are_collected(api: RegistryAPI) -> None:
    by_id = {op.operation_id: op for op in api.operations}
    create = by_id["createSession"]
    assert "201" in create.examples
    assert create.examples["201"]["sessionId"] == "1234-abcd"


def test_parse_openapi_accepts_a_mapping() -> None:
    mapping = {
        "openapi": "3.0.3",
        "info": {"title": "Tiny API", "version": "1.0.0"},
        "paths": {
            "/ping": {
                "get": {
                    "operationId": "ping",
                    "responses": {"200": {"description": "ok"}},
                }
            }
        },
    }
    api_from_mapping = parse_openapi(mapping)
    assert api_from_mapping.name == "tiny-api"
    assert api_from_mapping.operations[0].operation_id == "ping"


def test_missing_operation_id_warns_and_synthesises() -> None:
    mapping = {
        "openapi": "3.0.3",
        "info": {"title": "Tiny API", "version": "1.0.0"},
        "paths": {
            "/ping": {"get": {"responses": {"200": {"description": "ok"}}}}
        },
    }
    parser = OpenAPIParser()
    result = parser.parse(mapping)
    assert result.operations[0].operation_id == "get__ping"
    assert parser.warnings


def test_strict_mode_raises_on_warning() -> None:
    mapping = {
        "openapi": "3.0.3",
        "info": {"title": "Tiny API", "version": "1.0.0"},
        "paths": {
            "/ping": {"get": {"responses": {"200": {"description": "ok"}}}}
        },
    }
    with pytest.raises(OpenAPIParseError):
        OpenAPIParser(strict=True).parse(mapping)


def test_swagger_2_is_rejected() -> None:
    with pytest.raises(OpenAPIParseError):
        parse_openapi({"swagger": "2.0", "info": {"title": "x", "version": "1"}, "paths": {}})


def test_build_registry_feeds_the_registry() -> None:
    registry = build_registry([FIXTURE], version="0.1.0")
    assert isinstance(registry, CamaraRegistry)
    assert registry.has_api("quality-on-demand")
    op = registry.require_operation("createSession")
    assert op.method is HTTPMethod.POST
