import pytest

from app.backends.acestep_backend import AceStepBackend


def test_raises_clear_error_when_acestep_not_installed():
    # This test assumes the optional `acestep` dependency is NOT installed
    # in the local/CI environment (it lives in the `kaggle` extra only).
    with pytest.raises(RuntimeError, match="acestep"):
        AceStepBackend()
