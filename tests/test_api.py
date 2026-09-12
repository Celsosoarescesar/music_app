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
