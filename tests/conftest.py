import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("MUSIC_BACKEND", "mock")
    with TestClient(app) as test_client:
        yield test_client
