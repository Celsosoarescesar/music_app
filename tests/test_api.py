import io

import numpy as np
import soundfile as sf

from app.utils import audio_array_to_wav_buffer


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
