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
