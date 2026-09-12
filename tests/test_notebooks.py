import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/ace_step_api_kaggle.ipynb")


def test_notebook_is_valid_json():
    content = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert content["nbformat"] == 4


def test_notebook_has_expected_cell_markers():
    content = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    sources = [
        "".join(cell["source"])
        for cell in content["cells"]
        if cell["cell_type"] == "code"
    ]
    joined = "\n".join(sources)
    assert "uv sync" in joined
    assert "MUSIC_BACKEND" in joined
    assert "ngrok" in joined
    assert "uvicorn" in joined
