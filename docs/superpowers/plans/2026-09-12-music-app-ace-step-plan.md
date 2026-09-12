# music_app ACE-Step API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI service (with a swappable mock/ACE-Step backend) exposing 6 music-generation inference endpoints, testable locally without a GPU and runnable for real inside a Kaggle notebook.

**Architecture:** A single FastAPI app (`app/main.py`) picks a `MusicBackend` implementation at startup based on `MUSIC_BACKEND` (`mock` or `acestep`). Both backends implement the same 6-method protocol, so routes never branch on backend type. Requests are validated with Pydantic models before reaching the backend; responses are streamed back as `audio/wav`.

**Tech Stack:** Python >=3.10, FastAPI, Uvicorn, Pydantic v2, pydantic-settings, NumPy, soundfile, python-multipart, uv (dependency management), pytest + httpx (tests). Optional (Kaggle-only): `acestep`, `pyngrok`.

**Spec:** `docs/superpowers/specs/2026-09-12-music-app-ace-step-design.md`

## Global Constraints

- Python `>=3.10`.
- Dependency management via `uv` (`pyproject.toml`); base deps always installed, `acestep` + `pyngrok` live in an optional `kaggle` dependency group, never required locally.
- `MUSIC_BACKEND` env var selects the backend: `mock` (default, no GPU, no `acestep` install needed) or `acestep` (real model, Kaggle only).
- All 6 endpoints are synchronous (the HTTP request blocks until the audio is ready) and return `audio/wav` via `StreamingResponse`.
- `duration` (and any generated audio length) is always an explicit positive value — this project never passes through the model's `-1 = random duration` convenience option.
- No stem separation, no voice cloning, no LoRA training anywhere in this plan (documented out-of-scope in the spec).
- The automated test suite must pass on a machine with no GPU and without the `acestep` package installed — it only exercises the mock backend, plus a guard-clause check on the real backend.

---

## Task 1: Project scaffolding & settings

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `app/__init__.py`
- Create: `app/settings.py`
- Test: `tests/test_settings.py`

**Interfaces:**
- Produces: `app.settings.AppSettings` — a `pydantic_settings.BaseSettings` subclass with fields `music_backend: Literal["mock", "acestep"]` (default `"mock"`), `port: int` (default `8000`), `ace_step_checkpoint_path: str | None` (default `None`). Later tasks construct it as `AppSettings()` (reads env/`.env`) or `AppSettings(_env_file=None)` (env vars only, ignores any `.env` file — used by tests for isolation).

- [ ] **Step 1: Confirm `uv` is available**

Run: `uv --version`
Expected: prints a version string. If the command is not found, install it first with `pip install uv` and re-run.

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "music-app"
version = "0.1.0"
description = "Projeto de teste da API FastAPI para o modelo de geracao de musica ACE-Step"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.115",
    "uvicorn>=0.32",
    "pydantic>=2.9",
    "pydantic-settings>=2.6",
    "numpy>=1.26",
    "soundfile>=0.12",
    "python-multipart>=0.0.12",
]

[project.optional-dependencies]
kaggle = [
    "acestep @ git+https://github.com/ace-step/ACE-Step.git",
    "pyngrok>=7.2",
]

[dependency-groups]
dev = [
    "pytest>=8.3",
    "httpx>=0.27",
]

[tool.pytest.ini_options]
testpaths = ["tests"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["app"]
```

- [ ] **Step 3: Create `.gitignore`**

```
__pycache__/
*.py[cod]
.venv/
.env
usage.csv
.pytest_cache/
*.egg-info/
.ipynb_checkpoints/
```

- [ ] **Step 4: Create `.env.example`**

```
MUSIC_BACKEND=mock
PORT=8000
ACE_STEP_CHECKPOINT_PATH=
```

- [ ] **Step 5: Create empty `app/__init__.py`**

```python
```

- [ ] **Step 6: Write the failing test for settings**

Create `tests/test_settings.py`:

```python
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
```

- [ ] **Step 7: Run the test to verify it fails**

Run: `uv run pytest tests/test_settings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.settings'` (or similar import error).

- [ ] **Step 8: Implement `app/settings.py`**

```python
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    music_backend: Literal["mock", "acestep"] = "mock"
    port: int = 8000
    ace_step_checkpoint_path: str | None = None
```

- [ ] **Step 9: Sync dependencies and run the test to verify it passes**

Run: `uv sync`
Run: `uv run pytest tests/test_settings.py -v`
Expected: 3 passed.

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml .gitignore .env.example app/__init__.py app/settings.py tests/test_settings.py uv.lock
git commit -m "feat: scaffold project and add AppSettings"
```

---

## Task 2: Audio buffer utilities

**Files:**
- Create: `app/utils.py`
- Test: `tests/test_utils.py`

**Interfaces:**
- Consumes: none (only `numpy` and `soundfile`).
- Produces:
  - `app.utils.audio_array_to_wav_buffer(audio: np.ndarray, sample_rate: int) -> io.BytesIO` — a seek-to-0 buffer containing a valid WAV file.
  - `app.utils.read_wav_upload(data: bytes) -> tuple[np.ndarray, int]` — decodes WAV bytes into `(audio_array, sample_rate)`; raises `ValueError` on invalid/corrupted input.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_utils.py`:

```python
import numpy as np
import pytest

from app.utils import audio_array_to_wav_buffer, read_wav_upload


def test_round_trip_preserves_duration():
    sample_rate = 32000
    duration_seconds = 1.5
    audio = np.zeros(int(sample_rate * duration_seconds), dtype=np.float32)

    buffer = audio_array_to_wav_buffer(audio, sample_rate)
    decoded_audio, decoded_sample_rate = read_wav_upload(buffer.read())

    assert decoded_sample_rate == sample_rate
    assert abs(len(decoded_audio) / decoded_sample_rate - duration_seconds) < 0.01


def test_read_wav_upload_rejects_garbage_bytes():
    with pytest.raises(ValueError):
        read_wav_upload(b"this is not a wav file")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_utils.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.utils'`.

- [ ] **Step 3: Implement `app/utils.py`**

```python
from io import BytesIO

import numpy as np
import soundfile as sf


def audio_array_to_wav_buffer(audio: np.ndarray, sample_rate: int) -> BytesIO:
    buffer = BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer


def read_wav_upload(data: bytes) -> tuple[np.ndarray, int]:
    try:
        audio, sample_rate = sf.read(BytesIO(data), dtype="float32")
    except Exception as exc:
        raise ValueError(f"Could not decode audio file: {exc}") from exc
    return audio, sample_rate
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_utils.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add app/utils.py tests/test_utils.py
git commit -m "feat: add WAV buffer encode/decode utilities"
```

---

## Task 3: Backend protocol & mock backend

**Files:**
- Create: `app/backends/__init__.py`
- Create: `app/backends/base.py`
- Create: `app/backends/mock_backend.py`
- Test: `tests/test_mock_backend.py`

**Interfaces:**
- Produces:
  - `app.backends.base.EditMode` — `Literal["only_lyrics", "remix"]`.
  - `app.backends.base.MusicBackend` — a `Protocol` with 6 methods (exact signatures below); every backend in this project implements all 6.
  - `app.backends.mock_backend.MockMusicBackend` — a concrete `MusicBackend` implementation using a synthetic sine tone; `.SAMPLE_RATE` class attribute (`int`) exposes the sample rate it generates at.

- [ ] **Step 1: Create empty `app/backends/__init__.py`**

```python
```

- [ ] **Step 2: Write `app/backends/base.py`**

```python
from typing import Literal, Protocol

import numpy as np

EditMode = Literal["only_lyrics", "remix"]


class MusicBackend(Protocol):
    def text2music(
        self,
        tags: str,
        lyrics: str,
        duration: float,
        seed: int | None,
        steps: int,
        guidance_scale: float,
    ) -> tuple[np.ndarray, int]: ...

    def retake(
        self,
        tags: str,
        lyrics: str,
        duration: float,
        seed: int | None,
        steps: int,
        guidance_scale: float,
        variance: float,
    ) -> tuple[np.ndarray, int]: ...

    def repaint(
        self,
        audio: np.ndarray,
        sample_rate: int,
        start_time: float,
        end_time: float,
        tags: str,
        lyrics: str,
    ) -> tuple[np.ndarray, int]: ...

    def edit(
        self,
        audio: np.ndarray,
        sample_rate: int,
        tags: str,
        lyrics: str,
        mode: EditMode,
    ) -> tuple[np.ndarray, int]: ...

    def extend(
        self,
        audio: np.ndarray,
        sample_rate: int,
        left_extend_seconds: float,
        right_extend_seconds: float,
        tags: str,
        lyrics: str,
    ) -> tuple[np.ndarray, int]: ...

    def audio2audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        tags: str,
        lyrics: str,
    ) -> tuple[np.ndarray, int]: ...
```

This file has no tests of its own — a `Protocol` carries no runtime behavior. It is exercised indirectly by every backend that implements it.

- [ ] **Step 3: Write the failing tests for the mock backend**

Create `tests/test_mock_backend.py`:

```python
from app.backends.mock_backend import MockMusicBackend
from app.utils import audio_array_to_wav_buffer


def _duration(audio, sample_rate) -> float:
    return len(audio) / float(sample_rate)


def test_text2music_returns_requested_duration():
    backend = MockMusicBackend()
    audio, sample_rate = backend.text2music(
        tags="lo-fi", lyrics="", duration=2.0, seed=None, steps=10, guidance_scale=5.0
    )
    assert abs(_duration(audio, sample_rate) - 2.0) < 0.01


def test_retake_returns_requested_duration():
    backend = MockMusicBackend()
    audio, sample_rate = backend.retake(
        tags="lo-fi", lyrics="", duration=1.5, seed=1, steps=10,
        guidance_scale=5.0, variance=0.3,
    )
    assert abs(_duration(audio, sample_rate) - 1.5) < 0.01


def test_repaint_keeps_input_duration():
    backend = MockMusicBackend()
    input_audio, input_sr = backend.text2music(
        tags="lo-fi", lyrics="", duration=3.0, seed=None, steps=10, guidance_scale=5.0
    )
    audio, sample_rate = backend.repaint(
        audio=input_audio, sample_rate=input_sr,
        start_time=1.0, end_time=2.0, tags="lo-fi", lyrics="",
    )
    assert abs(_duration(audio, sample_rate) - 3.0) < 0.01


def test_edit_keeps_input_duration():
    backend = MockMusicBackend()
    input_audio, input_sr = backend.text2music(
        tags="lo-fi", lyrics="", duration=2.5, seed=None, steps=10, guidance_scale=5.0
    )
    audio, sample_rate = backend.edit(
        audio=input_audio, sample_rate=input_sr,
        tags="lo-fi", lyrics="[verse]\nhi", mode="remix",
    )
    assert abs(_duration(audio, sample_rate) - 2.5) < 0.01


def test_extend_adds_requested_seconds():
    backend = MockMusicBackend()
    input_audio, input_sr = backend.text2music(
        tags="lo-fi", lyrics="", duration=2.0, seed=None, steps=10, guidance_scale=5.0
    )
    audio, sample_rate = backend.extend(
        audio=input_audio, sample_rate=input_sr,
        left_extend_seconds=0.5, right_extend_seconds=1.0,
        tags="lo-fi", lyrics="",
    )
    assert abs(_duration(audio, sample_rate) - 3.5) < 0.01


def test_audio2audio_keeps_reference_duration():
    backend = MockMusicBackend()
    input_audio, input_sr = backend.text2music(
        tags="lo-fi", lyrics="", duration=4.0, seed=None, steps=10, guidance_scale=5.0
    )
    audio, sample_rate = backend.audio2audio(
        audio=input_audio, sample_rate=input_sr, tags="lo-fi", lyrics="",
    )
    assert abs(_duration(audio, sample_rate) - 4.0) < 0.01


def test_mock_output_round_trips_through_wav_buffer():
    backend = MockMusicBackend()
    audio, sample_rate = backend.text2music(
        tags="lo-fi", lyrics="", duration=1.0, seed=None, steps=10, guidance_scale=5.0
    )
    buffer = audio_array_to_wav_buffer(audio, sample_rate)
    assert buffer.read(4) == b"RIFF"
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/test_mock_backend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.backends.mock_backend'`.

- [ ] **Step 5: Implement `app/backends/mock_backend.py`**

```python
import numpy as np

_TONE_HZ = 440.0


class MockMusicBackend:
    """Synthetic backend used for local development and tests, with no GPU
    and no `acestep` dependency required. Every method returns a sine tone
    whose duration matches what the real ACE-Step backend would be expected
    to produce for the same inputs — it validates the API's contract, not
    the model's output quality.
    """

    SAMPLE_RATE = 32000

    def _make_tone(self, duration_seconds: float) -> np.ndarray:
        num_samples = max(1, int(round(duration_seconds * self.SAMPLE_RATE)))
        t = np.linspace(0, duration_seconds, num_samples, endpoint=False)
        return (0.1 * np.sin(2 * np.pi * _TONE_HZ * t)).astype(np.float32)

    def text2music(self, tags, lyrics, duration, seed, steps, guidance_scale):
        return self._make_tone(duration), self.SAMPLE_RATE

    def retake(self, tags, lyrics, duration, seed, steps, guidance_scale, variance):
        return self._make_tone(duration), self.SAMPLE_RATE

    def repaint(self, audio, sample_rate, start_time, end_time, tags, lyrics):
        duration = len(audio) / float(sample_rate)
        return self._make_tone(duration), self.SAMPLE_RATE

    def edit(self, audio, sample_rate, tags, lyrics, mode):
        duration = len(audio) / float(sample_rate)
        return self._make_tone(duration), self.SAMPLE_RATE

    def extend(self, audio, sample_rate, left_extend_seconds, right_extend_seconds, tags, lyrics):
        duration = len(audio) / float(sample_rate) + left_extend_seconds + right_extend_seconds
        return self._make_tone(duration), self.SAMPLE_RATE

    def audio2audio(self, audio, sample_rate, tags, lyrics):
        duration = len(audio) / float(sample_rate)
        return self._make_tone(duration), self.SAMPLE_RATE
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_mock_backend.py -v`
Expected: 7 passed.

- [ ] **Step 7: Commit**

```bash
git add app/backends/__init__.py app/backends/base.py app/backends/mock_backend.py tests/test_mock_backend.py
git commit -m "feat: add MusicBackend protocol and mock backend"
```

---

## Task 4: Request schemas

**Files:**
- Create: `app/schemas.py`
- Test: `tests/test_schemas.py`

**Interfaces:**
- Consumes: `app.backends.base.EditMode` (Task 3).
- Produces (all `pydantic.BaseModel`):
  - `Text2MusicRequest(tags: str, lyrics: str = "", duration: float, seed: int | None = None, steps: int = 27, guidance_scale: float = 7.5)`
  - `RetakeRequest(Text2MusicRequest)` + `variance: float = 0.5`
  - `RepaintParams(start_time: float, end_time: float, tags: str, lyrics: str = "")` — raises if `start_time >= end_time`.
  - `EditParams(tags: str, lyrics: str = "", mode: EditMode = "only_lyrics")`
  - `ExtendParams(left_extend_seconds: float = 0.0, right_extend_seconds: float = 0.0, tags: str = "", lyrics: str = "")` — raises if both extend values are `<= 0`.
  - `Audio2AudioParams(tags: str, lyrics: str = "")`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_schemas.py`:

```python
import pytest
from pydantic import ValidationError

from app.schemas import (
    Audio2AudioParams,
    EditParams,
    ExtendParams,
    RepaintParams,
    RetakeRequest,
    Text2MusicRequest,
)


def test_text2music_request_accepts_valid_payload():
    request = Text2MusicRequest(tags="lo-fi, chill", duration=30.0)
    assert request.lyrics == ""
    assert request.steps == 27
    assert request.guidance_scale == 7.5


def test_text2music_request_rejects_non_positive_duration():
    with pytest.raises(ValidationError):
        Text2MusicRequest(tags="lo-fi", duration=0)


def test_text2music_request_rejects_empty_tags():
    with pytest.raises(ValidationError):
        Text2MusicRequest(tags="", duration=10.0)


def test_retake_request_inherits_text2music_fields():
    request = RetakeRequest(tags="lo-fi", duration=10.0, variance=0.2)
    assert request.variance == 0.2
    assert request.steps == 27


def test_repaint_params_rejects_start_after_end():
    with pytest.raises(ValidationError):
        RepaintParams(start_time=5.0, end_time=1.0, tags="lo-fi")


def test_repaint_params_accepts_valid_range():
    params = RepaintParams(start_time=1.0, end_time=5.0, tags="lo-fi")
    assert params.lyrics == ""


def test_edit_params_rejects_invalid_mode():
    with pytest.raises(ValidationError):
        EditParams(tags="lo-fi", mode="not-a-mode")


def test_edit_params_accepts_valid_mode():
    params = EditParams(tags="lo-fi", mode="remix")
    assert params.mode == "remix"


def test_extend_params_rejects_zero_extension_on_both_sides():
    with pytest.raises(ValidationError):
        ExtendParams(left_extend_seconds=0, right_extend_seconds=0, tags="lo-fi")


def test_extend_params_accepts_one_sided_extension():
    params = ExtendParams(left_extend_seconds=0, right_extend_seconds=2.0, tags="lo-fi")
    assert params.right_extend_seconds == 2.0


def test_audio2audio_params_accepts_valid_payload():
    params = Audio2AudioParams(tags="lo-fi")
    assert params.lyrics == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_schemas.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.schemas'`.

- [ ] **Step 3: Implement `app/schemas.py`**

```python
from pydantic import BaseModel, Field, model_validator

from app.backends.base import EditMode


class Text2MusicRequest(BaseModel):
    tags: str = Field(min_length=1, max_length=1000)
    lyrics: str = Field(default="", max_length=5000)
    duration: float = Field(gt=0, le=600)
    seed: int | None = None
    steps: int = Field(default=27, ge=1, le=200)
    guidance_scale: float = Field(default=7.5, ge=0.0, le=20.0)


class RetakeRequest(Text2MusicRequest):
    variance: float = Field(default=0.5, ge=0.0, le=1.0)


class RepaintParams(BaseModel):
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    tags: str = Field(min_length=1, max_length=1000)
    lyrics: str = Field(default="", max_length=5000)

    @model_validator(mode="after")
    def check_time_range(self) -> "RepaintParams":
        if self.start_time >= self.end_time:
            raise ValueError("start_time must be less than end_time")
        return self


class EditParams(BaseModel):
    tags: str = Field(min_length=1, max_length=1000)
    lyrics: str = Field(default="", max_length=5000)
    mode: EditMode = "only_lyrics"


class ExtendParams(BaseModel):
    left_extend_seconds: float = Field(default=0.0, ge=0.0, le=300.0)
    right_extend_seconds: float = Field(default=0.0, ge=0.0, le=300.0)
    tags: str = Field(default="", max_length=1000)
    lyrics: str = Field(default="", max_length=5000)

    @model_validator(mode="after")
    def check_at_least_one_side(self) -> "ExtendParams":
        if self.left_extend_seconds <= 0 and self.right_extend_seconds <= 0:
            raise ValueError(
                "at least one of left_extend_seconds/right_extend_seconds must be > 0"
            )
        return self


class Audio2AudioParams(BaseModel):
    tags: str = Field(min_length=1, max_length=1000)
    lyrics: str = Field(default="", max_length=5000)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_schemas.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add app/schemas.py tests/test_schemas.py
git commit -m "feat: add Pydantic request schemas for all 6 endpoints"
```

---

## Task 5: ACE-Step backend stub

**Files:**
- Create: `app/backends/acestep_backend.py`
- Test: `tests/test_acestep_backend.py`

**Interfaces:**
- Consumes: `app.backends.base.MusicBackend`, `app.backends.base.EditMode` (Task 3).
- Produces: `app.backends.acestep_backend.AceStepBackend(checkpoint_path: str | None = None)` — implements all 6 `MusicBackend` methods by delegating to the real `acestep` package; raises `RuntimeError` on construction if `acestep` is not installed.

This backend is only ever exercised for real inside the Kaggle notebook (Task 10), where the `acestep` package is installed. The local test suite only checks the guard clause below — it must not require a GPU or the `acestep` package.

- [ ] **Step 1: Write the failing test**

Create `tests/test_acestep_backend.py`:

```python
import pytest

from app.backends.acestep_backend import AceStepBackend


def test_raises_clear_error_when_acestep_not_installed():
    # This test assumes the optional `acestep` dependency is NOT installed
    # in the local/CI environment (it lives in the `kaggle` extra only).
    with pytest.raises(RuntimeError, match="acestep"):
        AceStepBackend()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_acestep_backend.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.backends.acestep_backend'`.

- [ ] **Step 3: Implement `app/backends/acestep_backend.py`**

```python
"""Real ACE-Step backend.

Only usable in an environment where the optional `acestep` dependency is
installed (see the `kaggle` extra in pyproject.toml) — typically inside the
Kaggle notebook at `notebooks/ace_step_api_kaggle.ipynb`. Not exercised by
the local/mock test suite beyond the "not installed" guard clause.

NOTE FOR IMPLEMENTERS: the public ACE-Step documentation describes its
parameters conceptually (tags/prompt, lyrics, duration, steps,
guidance_scale, seed, and per-task inputs for retake/repaint/edit/extend/
audio2audio) but does not pin an exact, versioned Python API. Before
relying on this file with a real checkpoint, check the signature of the
installed package (`python -c "import acestep; help(acestep.ACEStep)"` or
read its source under `site-packages/acestep/`) and adjust the calls below
if they differ.
"""

from __future__ import annotations

from app.backends.base import EditMode


class AceStepBackend:
    def __init__(self, checkpoint_path: str | None = None) -> None:
        try:
            from acestep import ACEStep
        except ImportError as exc:
            raise RuntimeError(
                "The 'acestep' package is not installed. Install it with "
                "`uv sync --extra kaggle` (or "
                "`pip install git+https://github.com/ace-step/ACE-Step.git`) "
                "before using MUSIC_BACKEND=acestep."
            ) from exc
        self._model = ACEStep(checkpoint_path=checkpoint_path)

    def text2music(self, tags, lyrics, duration, seed, steps, guidance_scale):
        audio = self._model.generate(
            prompt=tags,
            lyrics=lyrics,
            duration=duration,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return audio, self._model.sample_rate

    def retake(self, tags, lyrics, duration, seed, steps, guidance_scale, variance):
        audio = self._model.generate(
            prompt=tags,
            lyrics=lyrics,
            duration=duration,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            variance=variance,
            task="retake",
        )
        return audio, self._model.sample_rate

    def repaint(self, audio, sample_rate, start_time, end_time, tags, lyrics):
        output = self._model.generate(
            prompt=tags,
            lyrics=lyrics,
            task="repaint",
            reference_audio=audio,
            reference_sample_rate=sample_rate,
            start_time=start_time,
            end_time=end_time,
        )
        return output, self._model.sample_rate

    def edit(self, audio, sample_rate, tags, lyrics, mode: EditMode):
        output = self._model.generate(
            prompt=tags,
            lyrics=lyrics,
            task="edit",
            edit_mode=mode,
            reference_audio=audio,
            reference_sample_rate=sample_rate,
        )
        return output, self._model.sample_rate

    def extend(self, audio, sample_rate, left_extend_seconds, right_extend_seconds, tags, lyrics):
        output = self._model.generate(
            prompt=tags,
            lyrics=lyrics,
            task="extend",
            reference_audio=audio,
            reference_sample_rate=sample_rate,
            left_extend_seconds=left_extend_seconds,
            right_extend_seconds=right_extend_seconds,
        )
        return output, self._model.sample_rate

    def audio2audio(self, audio, sample_rate, tags, lyrics):
        output = self._model.generate(
            prompt=tags,
            lyrics=lyrics,
            task="audio2audio",
            reference_audio=audio,
            reference_sample_rate=sample_rate,
        )
        return output, self._model.sample_rate
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_acestep_backend.py -v`
Expected: 1 passed (fails instead with a mismatched error if `acestep` happens to be installed in this environment — it should not be, per Global Constraints).

- [ ] **Step 5: Commit**

```bash
git add app/backends/acestep_backend.py tests/test_acestep_backend.py
git commit -m "feat: add ACE-Step backend with install guard clause"
```

---

## Task 6: FastAPI app skeleton (lifespan, health check, usage logging)

**Files:**
- Create: `app/main.py`
- Create: `tests/conftest.py`
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: `app.settings.AppSettings` (Task 1), `app.backends.base.MusicBackend` (Task 3), `app.backends.mock_backend.MockMusicBackend` (Task 3), `app.backends.acestep_backend.AceStepBackend` (Task 5, imported lazily).
- Produces:
  - `app.main.app` — the FastAPI instance. On startup, `app.state.settings` holds the loaded `AppSettings` and `app.state.backend` holds the selected `MusicBackend` instance.
  - `tests/conftest.py` fixture `client` — a `fastapi.testclient.TestClient` wrapping `app.main.app`, with `MUSIC_BACKEND` forced to `"mock"` before startup. Reused by every later test file that needs to call the API.

- [ ] **Step 1: Write the shared test fixture**

Create `tests/conftest.py`:

```python
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("MUSIC_BACKEND", "mock")
    with TestClient(app) as test_client:
        yield test_client
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_main.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_main.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.main'`.

- [ ] **Step 4: Implement `app/main.py` (skeleton only — endpoints added in Tasks 7-8)**

```python
import csv
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator, Awaitable, Callable
from uuid import uuid4

from fastapi import FastAPI, Request, Response

from app.backends.base import MusicBackend
from app.backends.mock_backend import MockMusicBackend
from app.settings import AppSettings

USAGE_LOG_PATH = "usage.csv"
USAGE_LOG_HEADER = [
    "request_id",
    "datetime",
    "endpoint",
    "response_time_seconds",
    "status_code",
]


def _load_backend(settings: AppSettings) -> MusicBackend:
    if settings.music_backend == "mock":
        return MockMusicBackend()
    if settings.music_backend == "acestep":
        from app.backends.acestep_backend import AceStepBackend

        return AceStepBackend(checkpoint_path=settings.ace_step_checkpoint_path)
    raise ValueError(f"Unknown MUSIC_BACKEND: {settings.music_backend}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = AppSettings()
    app.state.settings = settings
    app.state.backend = _load_backend(settings)
    yield
    app.state.backend = None


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def log_usage(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    request_id = uuid4().hex
    request_datetime = datetime.now(timezone.utc).isoformat()
    start_time = time.perf_counter()
    response = await call_next(request)
    response_time = round(time.perf_counter() - start_time, 4)
    response.headers["X-API-Request-ID"] = request_id
    response.headers["X-Response-Time"] = str(response_time)
    with open(USAGE_LOG_PATH, "a", newline="") as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(USAGE_LOG_HEADER)
        writer.writerow(
            [
                request_id,
                request_datetime,
                request.url.path,
                response_time,
                response.status_code,
            ]
        )
    return response


@app.get("/")
def health_check() -> dict:
    return {"status": "healthy"}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_main.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add app/main.py tests/conftest.py tests/test_main.py
git commit -m "feat: add FastAPI app skeleton with lifespan and usage logging"
```

---

## Task 7: `text2music` and `retake` endpoints

**Files:**
- Modify: `app/main.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `app.schemas.Text2MusicRequest`, `app.schemas.RetakeRequest` (Task 4); `app.utils.audio_array_to_wav_buffer` (Task 2); `MusicBackend.text2music`/`.retake` (Task 3).
- Produces: routes `POST /generate/text2music`, `POST /generate/retake` on `app.main.app`; helper `app.main._audio_response(audio, sample_rate) -> StreamingResponse` reused by Task 8.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_api.py`:

```python
import io

import soundfile as sf


def test_text2music_returns_wav_with_requested_duration(client):
    response = client.post(
        "/generate/text2music",
        json={"tags": "lo-fi, chill", "duration": 2.0, "steps": 10, "guidance_scale": 5.0},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    audio, sample_rate = sf.read(io.BytesIO(response.content))
    assert abs(len(audio) / sample_rate - 2.0) < 0.01


def test_text2music_rejects_non_positive_duration(client):
    response = client.post(
        "/generate/text2music", json={"tags": "lo-fi", "duration": 0}
    )
    assert response.status_code == 422


def test_retake_returns_wav_with_requested_duration(client):
    response = client.post(
        "/generate/retake",
        json={"tags": "lo-fi, chill", "duration": 1.5, "variance": 0.3},
    )
    assert response.status_code == 200
    audio, sample_rate = sf.read(io.BytesIO(response.content))
    assert abs(len(audio) / sample_rate - 1.5) < 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_api.py -v`
Expected: FAIL with 404 (routes don't exist yet — assertions on status_code 200 fail).

- [ ] **Step 3: Append the endpoints to `app/main.py`**

Add these imports at the top (alongside the existing ones):

```python
from fastapi.responses import StreamingResponse

from app.schemas import RetakeRequest, Text2MusicRequest
from app.utils import audio_array_to_wav_buffer
```

Add at the end of `app/main.py`:

```python
def _audio_response(audio, sample_rate: int) -> StreamingResponse:
    buffer = audio_array_to_wav_buffer(audio, sample_rate)
    return StreamingResponse(buffer, media_type="audio/wav")


@app.post("/generate/text2music", response_class=StreamingResponse)
def generate_text2music(payload: Text2MusicRequest, request: Request) -> StreamingResponse:
    backend: MusicBackend = request.app.state.backend
    audio, sample_rate = backend.text2music(
        tags=payload.tags,
        lyrics=payload.lyrics,
        duration=payload.duration,
        seed=payload.seed,
        steps=payload.steps,
        guidance_scale=payload.guidance_scale,
    )
    return _audio_response(audio, sample_rate)


@app.post("/generate/retake", response_class=StreamingResponse)
def generate_retake(payload: RetakeRequest, request: Request) -> StreamingResponse:
    backend: MusicBackend = request.app.state.backend
    audio, sample_rate = backend.retake(
        tags=payload.tags,
        lyrics=payload.lyrics,
        duration=payload.duration,
        seed=payload.seed,
        steps=payload.steps,
        guidance_scale=payload.guidance_scale,
        variance=payload.variance,
    )
    return _audio_response(audio, sample_rate)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_api.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_api.py
git commit -m "feat: add text2music and retake endpoints"
```

---

## Task 8: `repaint`, `edit`, `extend`, `audio2audio` endpoints

**Files:**
- Modify: `app/main.py`
- Modify: `tests/test_api.py`

**Interfaces:**
- Consumes: `app.schemas.RepaintParams`, `EditParams`, `ExtendParams`, `Audio2AudioParams` (Task 4); `app.utils.read_wav_upload`, `audio_array_to_wav_buffer` (Task 2); `MusicBackend.repaint`/`.edit`/`.extend`/`.audio2audio` (Task 3); `app.main._audio_response` (Task 7).
- Produces: routes `POST /generate/repaint`, `POST /generate/edit`, `POST /generate/extend`, `POST /generate/audio2audio`.

- [ ] **Step 1: Append the failing tests to `tests/test_api.py`**

```python
import numpy as np

from app.utils import audio_array_to_wav_buffer


def _wav_upload_bytes(duration_seconds: float, sample_rate: int = 32000) -> bytes:
    audio = np.zeros(int(duration_seconds * sample_rate), dtype=np.float32)
    return audio_array_to_wav_buffer(audio, sample_rate).read()


def test_repaint_keeps_original_duration(client):
    upload = _wav_upload_bytes(duration_seconds=3.0)
    response = client.post(
        "/generate/repaint",
        data={"start_time": "1.0", "end_time": "2.0", "tags": "lo-fi", "lyrics": ""},
        files={"audio_file": ("input.wav", upload, "audio/wav")},
    )
    assert response.status_code == 200
    audio, sample_rate = sf.read(io.BytesIO(response.content))
    assert abs(len(audio) / sample_rate - 3.0) < 0.01


def test_repaint_rejects_start_after_end(client):
    upload = _wav_upload_bytes(duration_seconds=3.0)
    response = client.post(
        "/generate/repaint",
        data={"start_time": "2.0", "end_time": "1.0", "tags": "lo-fi", "lyrics": ""},
        files={"audio_file": ("input.wav", upload, "audio/wav")},
    )
    assert response.status_code == 422


def test_repaint_rejects_corrupted_audio_file(client):
    response = client.post(
        "/generate/repaint",
        data={"start_time": "0", "end_time": "1", "tags": "lo-fi", "lyrics": ""},
        files={"audio_file": ("input.wav", b"not-a-real-wav-file", "audio/wav")},
    )
    assert response.status_code == 400


def test_edit_keeps_original_duration(client):
    upload = _wav_upload_bytes(duration_seconds=2.5)
    response = client.post(
        "/generate/edit",
        data={"tags": "lo-fi", "lyrics": "[verse]\nhi", "mode": "remix"},
        files={"audio_file": ("input.wav", upload, "audio/wav")},
    )
    assert response.status_code == 200
    audio, sample_rate = sf.read(io.BytesIO(response.content))
    assert abs(len(audio) / sample_rate - 2.5) < 0.01


def test_edit_rejects_invalid_mode(client):
    upload = _wav_upload_bytes(duration_seconds=2.5)
    response = client.post(
        "/generate/edit",
        data={"tags": "lo-fi", "lyrics": "", "mode": "not-a-mode"},
        files={"audio_file": ("input.wav", upload, "audio/wav")},
    )
    assert response.status_code == 422


def test_extend_adds_requested_seconds(client):
    upload = _wav_upload_bytes(duration_seconds=2.0)
    response = client.post(
        "/generate/extend",
        data={
            "left_extend_seconds": "0.5",
            "right_extend_seconds": "1.0",
            "tags": "lo-fi",
            "lyrics": "",
        },
        files={"audio_file": ("input.wav", upload, "audio/wav")},
    )
    assert response.status_code == 200
    audio, sample_rate = sf.read(io.BytesIO(response.content))
    assert abs(len(audio) / sample_rate - 3.5) < 0.01


def test_extend_rejects_zero_extension(client):
    upload = _wav_upload_bytes(duration_seconds=2.0)
    response = client.post(
        "/generate/extend",
        data={
            "left_extend_seconds": "0",
            "right_extend_seconds": "0",
            "tags": "lo-fi",
            "lyrics": "",
        },
        files={"audio_file": ("input.wav", upload, "audio/wav")},
    )
    assert response.status_code == 422


def test_audio2audio_keeps_reference_duration(client):
    upload = _wav_upload_bytes(duration_seconds=4.0)
    response = client.post(
        "/generate/audio2audio",
        data={"tags": "lo-fi", "lyrics": ""},
        files={"audio_file": ("input.wav", upload, "audio/wav")},
    )
    assert response.status_code == 200
    audio, sample_rate = sf.read(io.BytesIO(response.content))
    assert abs(len(audio) / sample_rate - 4.0) < 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_api.py -v`
Expected: FAIL with 404 for the 4 new routes.

- [ ] **Step 3: Append the endpoints to `app/main.py`**

Add these imports at the top (alongside the existing ones):

```python
from fastapi import File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from app.schemas import (
    Audio2AudioParams,
    EditParams,
    ExtendParams,
    RepaintParams,
)
from app.utils import read_wav_upload
```

Add at the end of `app/main.py`:

```python
def _read_upload_or_400(data: bytes):
    try:
        return read_wav_upload(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/generate/repaint", response_class=StreamingResponse)
async def generate_repaint(
    request: Request,
    audio_file: UploadFile = File(...),
    start_time: float = Form(...),
    end_time: float = Form(...),
    tags: str = Form(...),
    lyrics: str = Form(""),
) -> StreamingResponse:
    try:
        params = RepaintParams(
            start_time=start_time, end_time=end_time, tags=tags, lyrics=lyrics
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    audio, sample_rate = _read_upload_or_400(await audio_file.read())
    backend: MusicBackend = request.app.state.backend
    out_audio, out_sample_rate = backend.repaint(
        audio=audio,
        sample_rate=sample_rate,
        start_time=params.start_time,
        end_time=params.end_time,
        tags=params.tags,
        lyrics=params.lyrics,
    )
    return _audio_response(out_audio, out_sample_rate)


@app.post("/generate/edit", response_class=StreamingResponse)
async def generate_edit(
    request: Request,
    audio_file: UploadFile = File(...),
    tags: str = Form(...),
    lyrics: str = Form(""),
    mode: str = Form("only_lyrics"),
) -> StreamingResponse:
    try:
        params = EditParams(tags=tags, lyrics=lyrics, mode=mode)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    audio, sample_rate = _read_upload_or_400(await audio_file.read())
    backend: MusicBackend = request.app.state.backend
    out_audio, out_sample_rate = backend.edit(
        audio=audio,
        sample_rate=sample_rate,
        tags=params.tags,
        lyrics=params.lyrics,
        mode=params.mode,
    )
    return _audio_response(out_audio, out_sample_rate)


@app.post("/generate/extend", response_class=StreamingResponse)
async def generate_extend(
    request: Request,
    audio_file: UploadFile = File(...),
    left_extend_seconds: float = Form(0.0),
    right_extend_seconds: float = Form(0.0),
    tags: str = Form(""),
    lyrics: str = Form(""),
) -> StreamingResponse:
    try:
        params = ExtendParams(
            left_extend_seconds=left_extend_seconds,
            right_extend_seconds=right_extend_seconds,
            tags=tags,
            lyrics=lyrics,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    audio, sample_rate = _read_upload_or_400(await audio_file.read())
    backend: MusicBackend = request.app.state.backend
    out_audio, out_sample_rate = backend.extend(
        audio=audio,
        sample_rate=sample_rate,
        left_extend_seconds=params.left_extend_seconds,
        right_extend_seconds=params.right_extend_seconds,
        tags=params.tags,
        lyrics=params.lyrics,
    )
    return _audio_response(out_audio, out_sample_rate)


@app.post("/generate/audio2audio", response_class=StreamingResponse)
async def generate_audio2audio(
    request: Request,
    audio_file: UploadFile = File(...),
    tags: str = Form(...),
    lyrics: str = Form(""),
) -> StreamingResponse:
    try:
        params = Audio2AudioParams(tags=tags, lyrics=lyrics)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    audio, sample_rate = _read_upload_or_400(await audio_file.read())
    backend: MusicBackend = request.app.state.backend
    out_audio, out_sample_rate = backend.audio2audio(
        audio=audio, sample_rate=sample_rate, tags=params.tags, lyrics=params.lyrics
    )
    return _audio_response(out_audio, out_sample_rate)
```

`io` and `soundfile as sf` are already imported at the top of `tests/test_api.py` from Task 7 — no new imports needed for those two; only the `numpy` and `audio_array_to_wav_buffer` imports shown in Step 1 above are new.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_api.py -v`
Expected: 11 passed (3 from Task 7 + 8 new).

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest -v`
Expected: all tests across every file pass (Tasks 1-8 combined).

- [ ] **Step 6: Commit**

```bash
git add app/main.py tests/test_api.py
git commit -m "feat: add repaint, edit, extend, and audio2audio endpoints"
```

---

## Task 9: `CLAUDE.md` and `README.md`

**Files:**
- Create: `CLAUDE.md`
- Create: `README.md`

**Interfaces:** none (documentation only).

- [ ] **Step 1: Write `CLAUDE.md`**

```markdown
# music_app

## Project Overview

API FastAPI de teste para o modelo de geração de música ACE-Step
(https://github.com/ace-step/ACE-Step). Projeto de aprendizado: o objetivo
é aprender a operar o modelo (não é um produto). Roda em dois modos:

- `MUSIC_BACKEND=mock` (padrão): gera áudio sintético, sem GPU, sem o
  pacote `acestep` instalado. Usado no desenvolvimento local e nos testes.
- `MUSIC_BACKEND=acestep`: usa o modelo real. Requer GPU — pensado para
  rodar dentro de um notebook Kaggle (ver `notebooks/`).

## Tech Stack

- Python >=3.10, gerenciado com `uv`
- FastAPI + Uvicorn
- Pydantic v2 / pydantic-settings
- NumPy + soundfile (codificação/decodificação de WAV)
- pytest + httpx (testes)
- Opcional (grupo `kaggle`): `acestep`, `pyngrok`

## Project Structure

```
app/
├── main.py                # rotas FastAPI, lifespan, middleware de log
├── settings.py             # AppSettings (pydantic-settings)
├── schemas.py               # Pydantic request models por endpoint
├── backends/
│   ├── base.py               # Protocol MusicBackend (6 métodos)
│   ├── mock_backend.py       # backend sintético (sem GPU)
│   └── acestep_backend.py    # backend real (Kaggle)
└── utils.py                 # conversão array de áudio <-> buffer WAV
tests/                       # pytest, sempre roda com MUSIC_BACKEND=mock
notebooks/                   # notebook para subir a API real no Kaggle
docs/superpowers/            # spec e plano de implementação deste projeto
```

## Commands

- `uv sync` — instala as dependências base (modo mock/dev).
- `uv sync --extra kaggle` — inclui também `acestep` e `pyngrok` (só dentro
  do Kaggle).
- `uv run pytest -v` — roda a suíte de testes (sempre em modo mock).
- `uv run uvicorn app.main:app --reload` — sobe a API localmente em modo
  mock (padrão) em `http://127.0.0.1:8000`.
- `MUSIC_BACKEND=acestep uv run uvicorn app.main:app --host 0.0.0.0 --port 8000`
  — sobe a API com o modelo real (requer GPU e `acestep` instalado).

## Endpoints (todos síncronos, retornam `audio/wav`)

`POST /generate/text2music`, `/generate/retake`, `/generate/repaint`,
`/generate/edit`, `/generate/extend`, `/generate/audio2audio`. Veja
`docs/superpowers/specs/2026-09-12-music-app-ace-step-design.md` para os
campos de cada um.

## Known limitations (by design)

- Tom/BPM não são parâmetros estruturados — vão como texto livre em `tags`.
- Sem separação de stems (StemGen do ACE-Step ainda não foi lançado).
- Sem voice cloning (ambíguo entre inferência e treino de LoRA — fora do
  v1).
```

- [ ] **Step 2: Write `README.md`**

```markdown
# music_app

Projeto de teste para aprender a usar o modelo de geração de música
[ACE-Step](https://github.com/ace-step/ACE-Step) através de uma API
FastAPI.

## Rodando localmente (modo mock, sem GPU)

```bash
uv sync
uv run pytest -v
uv run uvicorn app.main:app --reload
```

Testando um endpoint (áudio sintético, só para validar o contrato da API):

```bash
curl -X POST http://127.0.0.1:8000/generate/text2music \
  -H "Content-Type: application/json" \
  -d '{"tags": "lo-fi, chill, piano", "lyrics": "", "duration": 10}' \
  --output saida.wav
```

## Rodando no Kaggle (modelo real, com GPU)

1. Faça upload de `notebooks/ace_step_api_kaggle.ipynb` num notebook Kaggle
   com acelerador de GPU ativado (T4 ou P100).
2. Rode as células na ordem — elas instalam as dependências (incluindo o
   `acestep`), sobem a API com `MUSIC_BACKEND=acestep` e abrem um túnel
   `ngrok` público.
3. Use a URL impressa pelo notebook para chamar os mesmos endpoints de
   fora do Kaggle (ex: com `curl` ou o cliente HTTP que preferir).

A sessão do Kaggle expira depois de algumas horas e a URL do `ngrok` muda a
cada nova sessão (plano gratuito) — normal para um projeto de teste.

## Documentação

- Spec de design: `docs/superpowers/specs/2026-09-12-music-app-ace-step-design.md`
- Plano de implementação: `docs/superpowers/plans/2026-09-12-music-app-ace-step-plan.md`
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: add CLAUDE.md and README.md"
```

---

## Task 10: Kaggle notebook

**Files:**
- Create: `notebooks/ace_step_api_kaggle.ipynb`
- Test: `tests/test_notebooks.py`

**Interfaces:** none (the notebook is not imported by application code).

- [ ] **Step 1: Write the failing test that checks the notebook is well-formed JSON with the expected cells**

Create `tests/test_notebooks.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_notebooks.py -v`
Expected: FAIL — `notebooks/ace_step_api_kaggle.ipynb` does not exist yet.

- [ ] **Step 3: Create `notebooks/ace_step_api_kaggle.ipynb`**

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# ACE-Step API no Kaggle\n",
    "\n",
    "Sobe a API FastAPI deste projeto usando o backend real (`MUSIC_BACKEND=acestep`)\n",
    "dentro de um notebook Kaggle com GPU, e expoe publicamente via ngrok.\n",
    "\n",
    "**Antes de rodar:** ative um acelerador de GPU em Notebook Settings (T4 ou P100),\n",
    "e tenha em maos um authtoken gratuito do ngrok (https://dashboard.ngrok.com/get-started/your-authtoken)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install -q uv\n",
    "!git clone https://github.com/SEU_USUARIO/music_app.git\n",
    "%cd music_app"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!uv sync --extra kaggle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "\n",
    "os.environ[\"MUSIC_BACKEND\"] = \"acestep\"\n",
    "os.environ[\"ACE_STEP_CHECKPOINT_PATH\"] = \"\"  # preencha se usar um checkpoint customizado\n",
    "os.environ[\"PORT\"] = \"8000\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import subprocess\n",
    "\n",
    "server_process = subprocess.Popen(\n",
    "    [\"uv\", \"run\", \"uvicorn\", \"app.main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"],\n",
    ")\n",
    "print(f\"uvicorn iniciado com PID {server_process.pid}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from pyngrok import ngrok, conf\n",
    "\n",
    "conf.get_default().auth_token = \"COLE_SEU_NGROK_AUTHTOKEN_AQUI\"\n",
    "public_tunnel = ngrok.connect(8000, \"http\")\n",
    "print(f\"API publica em: {public_tunnel.public_url}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Testando de fora do Kaggle\n",
    "\n",
    "Use a URL impressa acima no lugar de `http://127.0.0.1:8000` nos exemplos do `README.md`, por exemplo:\n",
    "\n",
    "```bash\n",
    "curl -X POST https://SUA-URL.ngrok-free.app/generate/text2music \\\n",
    "  -H \"Content-Type: application/json\" \\\n",
    "  -d '{\"tags\": \"lo-fi, chill, piano\", \"lyrics\": \"\", \"duration\": 10}' \\\n",
    "  --output saida.wav\n",
    "```"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Rode esta celula para encerrar o tunel e o servidor ao terminar a sessao.\n",
    "ngrok.disconnect(public_tunnel.public_url)\n",
    "server_process.terminate()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_notebooks.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add notebooks/ace_step_api_kaggle.ipynb tests/test_notebooks.py
git commit -m "feat: add Kaggle notebook to run the real ACE-Step backend"
```

---

## Task 11: Final verification

**Files:** none created or modified — this task only verifies the finished project.

- [ ] **Step 1: Run the complete test suite from a clean sync**

Run: `uv sync`
Run: `uv run pytest -v`
Expected: every test from Tasks 1-10 passes (settings, utils, mock backend, schemas, acestep backend guard clause, main/health, all 6 API endpoints, notebook check) — no failures, no skips.

- [ ] **Step 2: Smoke-test the running server manually**

Run: `uv run uvicorn app.main:app --port 8000` (in one terminal), then in another:

```bash
curl http://127.0.0.1:8000/
curl -X POST http://127.0.0.1:8000/generate/text2music \
  -H "Content-Type: application/json" \
  -d '{"tags": "lo-fi, chill", "duration": 3}' \
  --output /tmp/smoke_test.wav
```

Expected: the health check returns `{"status":"healthy"}`, and `/tmp/smoke_test.wav` is a playable ~3 second WAV file (mock tone). Stop the server (`Ctrl+C`) when done.

- [ ] **Step 3: Confirm git history is clean**

Run: `git log --oneline`
Expected: one commit per task (Tasks 1-10), all with descriptive messages, working tree clean (`git status` shows nothing to commit).

- [ ] **Step 4: Report completion**

No commit needed for this task — it is a verification gate, not a code change. If any step above fails, fix the underlying issue in the relevant earlier task's files before considering the plan complete.
