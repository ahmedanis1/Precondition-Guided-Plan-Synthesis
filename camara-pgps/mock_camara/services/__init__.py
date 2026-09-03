"""Mock CAMARA service routers, one module per API family."""

from mock_camara.services import location, network_insights, qod

__all__ = ["location", "qod", "network_insights"]
