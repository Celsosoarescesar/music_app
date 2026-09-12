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
