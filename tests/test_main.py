import csv
import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.backends.mock_backend import MockMusicBackend
from app.main import USAGE_LOG_PATH, log_usage


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


@pytest.fixture
def usage_log_cleanup():
    yield
    if os.path.exists(USAGE_LOG_PATH):
        os.remove(USAGE_LOG_PATH)


def test_log_usage_records_failed_requests(usage_log_cleanup):
    # Fresh, standalone app wired with the same log_usage middleware used by
    # app.main.app, so a raising route doesn't have to be added to the
    # shared production app just for this test.
    test_app = FastAPI()
    test_app.middleware("http")(log_usage)

    @test_app.get("/boom")
    def boom():
        raise RuntimeError("kaboom")

    test_client = TestClient(test_app)

    with pytest.raises(RuntimeError):
        test_client.get("/boom")

    with open(USAGE_LOG_PATH, newline="") as file:
        rows = list(csv.reader(file))
    last_row = rows[-1]
    assert last_row[2] == "/boom"
    assert last_row[4] == "500"
