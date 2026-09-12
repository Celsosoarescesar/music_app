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
