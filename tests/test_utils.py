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
