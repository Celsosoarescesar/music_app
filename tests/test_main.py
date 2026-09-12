from app.backends.mock_backend import MockMusicBackend


def test_health_check_returns_ok(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_lifespan_loads_mock_backend_by_default(client):
    assert isinstance(client.app.state.backend, MockMusicBackend)


def test_responses_carry_usage_headers(client):
    response = client.get("/")
    assert "X-API-Request-ID" in response.headers
    assert "X-Response-Time" in response.headers
