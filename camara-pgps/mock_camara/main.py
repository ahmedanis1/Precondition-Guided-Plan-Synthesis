"""Runnable mock CAMARA gateway.

    python -m uvicorn mock_camara.main:app --host 0.0.0.0 --port 8080
    mock-camara                                   # console script

Each service router is mounted twice: once at the bare operation path (what the
execution layer calls by default) and once behind the CAMARA-style
``/<api>/v0`` prefix. State is in-memory; ``POST /_mock/reset`` clears it.

This is a runnable execution environment, not the thesis evaluation.
"""

from __future__ import annotations

from fastapi import FastAPI

from mock_camara.services import location, network_insights, qod

_SERVICES = [
    (location.router, "/device-location/v0", None),
    (qod.router, "/quality-on-demand/v0", qod.reset),
    (network_insights.router, "/network-insights/v0", network_insights.reset),
]


def create_app() -> FastAPI:
    app = FastAPI(
        title="Mock CAMARA Gateway",
        version="0.1.0",
        summary="Local stand-in for Device Location, Quality on Demand and Network Insights.",
    )

    for router, prefix, _ in _SERVICES:
        # bare paths (what the executor calls) + CAMARA-style /<api>/v0 prefix
        app.include_router(router)
        app.include_router(router, prefix=prefix)

    @app.get("/health", tags=["meta"])
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "apis": ["device-location", "quality-on-demand", "network-insights"],
        }

    @app.post("/_mock/reset", status_code=204, tags=["meta"])
    def reset() -> None:
        for _, _, reset_fn in _SERVICES:
            if reset_fn is not None:
                reset_fn()

    return app


app = create_app()


def main() -> None:
    import uvicorn

    from camara_pgps.config import get_settings

    settings = get_settings()
    uvicorn.run(
        "mock_camara.main:app",
        host=settings.mock_camara.host,
        port=settings.mock_camara.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
