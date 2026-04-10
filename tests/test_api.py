from fastapi.testclient import TestClient
from api.main import app

test_client = TestClient(app)

def test_health_check():
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_decompose_endpoint():
    # Test our signal decomposition on Apple
    response = test_client.post("/api/v1/decompose", json={"symbol": "AAPL"})
    assert response.status_code == 200
    results = response.json()
    assert "symbol" in results
    assert "components" in results
    # The low-frequency trend should always be present
    assert "Long-term Trend" in results["components"]

def test_coherence_endpoint():
    # Test tracking resonance between Gold and S&P 500
    response = test_client.post("/api/v1/coherence", json={"first": "GLD", "second": "SPY"})
    assert response.status_code == 200
    results = response.json()
    assert "average_resonance" in results
    assert "map_shape" in results

def test_causality_endpoint():
    # Test if one asset leads another
    response = test_client.post("/api/v1/causality", json={"first": "GLD", "second": "SPY"})
    assert response.status_code == 200
    results = response.json()
    assert "causal_strength_fwd" in results

