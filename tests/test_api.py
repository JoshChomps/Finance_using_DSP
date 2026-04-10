import pytest
from fastapi.testclient import TestClient
from api.main import app
from engine.data import get_data

test_client = TestClient(app)

def test_health_check():
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def _data_available(*symbols):
    """Return True only when all symbols have cached/downloadable data."""
    return all(get_data(s) is not None for s in symbols)

def test_decompose_endpoint():
    if not _data_available("AAPL"):
        pytest.skip("AAPL data unavailable")
    response = test_client.post("/api/v1/decompose", json={"symbol": "AAPL"})
    assert response.status_code == 200
    results = response.json()
    assert "symbol" in results
    assert "components" in results
    assert "Long-term Trend" in results["components"]

def test_coherence_endpoint():
    if not _data_available("GLD", "SPY"):
        pytest.skip("GLD/SPY data unavailable")
    response = test_client.post("/api/v1/coherence", json={"first": "GLD", "second": "SPY"})
    assert response.status_code == 200
    results = response.json()
    assert "average_resonance" in results
    assert "map_shape" in results

def test_causality_endpoint():
    if not _data_available("GLD", "SPY"):
        pytest.skip("GLD/SPY data unavailable")
    response = test_client.post("/api/v1/causality", json={"first": "GLD", "second": "SPY"})
    assert response.status_code == 200
    results = response.json()
    assert "causal_strength_fwd" in results

