import pytest
from pydantic import ValidationError

from app.settings import AppSettings


def test_default_settings(monkeypatch):
    monkeypatch.delenv("MUSIC_BACKEND", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.delenv("ACE_STEP_CHECKPOINT_PATH", raising=False)

    settings = AppSettings(_env_file=None)

    assert settings.music_backend == "mock"
    assert settings.port == 8000
    assert settings.ace_step_checkpoint_path is None


def test_settings_read_from_env(monkeypatch):
    monkeypatch.setenv("MUSIC_BACKEND", "acestep")
    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setenv("ACE_STEP_CHECKPOINT_PATH", "/models/ace-step")

    settings = AppSettings(_env_file=None)

    assert settings.music_backend == "acestep"
    assert settings.port == 9000
    assert settings.ace_step_checkpoint_path == "/models/ace-step"


def test_invalid_backend_is_rejected(monkeypatch):
    monkeypatch.setenv("MUSIC_BACKEND", "not-a-backend")

    with pytest.raises(ValidationError):
        AppSettings(_env_file=None)
