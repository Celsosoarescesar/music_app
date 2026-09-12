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
