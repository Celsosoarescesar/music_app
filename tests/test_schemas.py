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
