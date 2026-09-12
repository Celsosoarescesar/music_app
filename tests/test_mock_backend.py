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
