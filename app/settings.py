from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    music_backend: Literal["mock", "acestep"] = "mock"
    port: int = 8000
    ace_step_checkpoint_path: str | None = None
