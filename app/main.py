import csv
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator, Awaitable, Callable
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from app.backends.base import MusicBackend
from app.backends.mock_backend import MockMusicBackend
from app.schemas import RetakeRequest, Text2MusicRequest
from app.settings import AppSettings
from app.utils import audio_array_to_wav_buffer

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
