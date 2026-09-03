"""Tests for the CLI (milestone 18)."""

from __future__ import annotations

import json

import pytest

from camara_pgps import cli
from camara_pgps.camara.registry import CamaraRegistry
from camara_pgps.config import PROJECT_ROOT
from camara_pgps.execution import MockExecutor
from camara_pgps.retrieval import HashingEmbedder, IngestionPipeline, Retriever, VectorStore
from camara_pgps.retrieval.rag import RagRetriever
from camara_pgps.workflow import CamaraWorkflow, WorkflowNodes
from scripts.run_agent import _to_run_argv

REGISTRY_FILE = PROJECT_ROOT / "data" / "registry" / "camara_apis.json"


@pytest.fixture
def stub_workflow(tmp_path) -> CamaraWorkflow:
    registry = CamaraRegistry.from_file(REGISTRY_FILE)
    embedder = HashingEmbedder()
    store = VectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="test_cli",
        distance="cosine",
    )
    IngestionPipeline(vector_store=store, embedder=embedder).ingest_registry(registry)
    rag = RagRetriever(
        retriever=Retriever(vector_store=store, embedder=embedder), registry=registry
    )
    return CamaraWorkflow(
        nodes=WorkflowNodes(
            registry=registry, rag_retriever=rag, executor=MockExecutor(registry=registry)
        )
    )


# --------------------------------------------------------------------------- #
# version / config
# --------------------------------------------------------------------------- #

def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert "camara-pgps 0.1.0" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["show-config"], ["--show-config"]])
def test_show_config(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(argv) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["environment"] == "development"


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == 0
    assert "usage:" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# list-apis
# --------------------------------------------------------------------------- #

def test_list_apis(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["list-apis"]) == 0
    out = capsys.readouterr().out
    assert "quality-on-demand" in out
    assert "createSession" in out


def test_list_apis_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["list-apis", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert {a["name"] for a in data} == {
        "device-location",
        "quality-on-demand",
        "network-insights",
    }


def test_list_apis_detail(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["list-apis", "--api", "quality-on-demand"]) == 0
    out = capsys.readouterr().out
    assert "createSession" in out
    assert "precondition:" in out
    assert "requires ['auth.token']" in out


def test_list_apis_unknown(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["list-apis", "--api", "nope"]) == 2
    assert "unknown API" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# ingest-camara
# --------------------------------------------------------------------------- #

def test_ingest_camara(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CAMARA_PGPS_EMBEDDING__PROVIDER", "hash")
    monkeypatch.setenv(
        "CAMARA_PGPS_VECTOR_STORE__PERSIST_DIRECTORY", str(tmp_path / "chroma")
    )
    monkeypatch.setenv("CAMARA_PGPS_VECTOR_STORE__COLLECTION_NAME", "cli_ingest")
    from camara_pgps.config import get_settings

    get_settings.cache_clear()

    assert cli.main(["ingest-camara", "--reset"]) == 0
    out = capsys.readouterr().out
    assert "ingested" in out and "chunk(s)" in out

    get_settings.cache_clear()


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #

def test_run_text_output(
    stub_workflow: CamaraWorkflow,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("camara_pgps.workflow.get_workflow", lambda: stub_workflow)
    code = cli.main(["run", "--request", "Improve the network quality for my video call"])
    assert code == 0
    out = capsys.readouterr().out
    assert "Understood the request as" in out


def test_run_json_output(
    stub_workflow: CamaraWorkflow,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("camara_pgps.workflow.get_workflow", lambda: stub_workflow)
    args = {
        "createSession": {
            "device": {"phoneNumber": "+12025550123"},
            "applicationServer": {"ipv4Address": "10.0.0.1"},
            "qosProfile": "QOS_L",
            "duration": 3600,
        }
    }
    code = cli.main(
        [
            "run",
            "--request",
            "Give my phone a QOS_L session for 1 hour",
            "--arguments",
            json.dumps(args),
            "--json",
        ]
    )
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["stage"] == "done"
    assert report["explanation"]
    assert [s["operation_id"] for s in report["plan"]["steps"]] == ["createSession"]


# --------------------------------------------------------------------------- #
# scripts/run_agent.py argv shaping
# --------------------------------------------------------------------------- #

def test_run_agent_argv_shaping() -> None:
    assert _to_run_argv(["improve my quality"]) == ["run", "--request", "improve my quality"]
    assert _to_run_argv(["--request", "x", "--json"]) == ["run", "--request", "x", "--json"]
    assert _to_run_argv([]) == ["run"]
