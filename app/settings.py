from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    music_backend: Literal["mock", "acestep"] = "mock"
    port: int = 8000
    ace_step_checkpoint_path: str | None = None

    @field_validator("ace_step_checkpoint_path", mode="before")
    @classmethod
    def _empty_string_to_none(cls, value: object) -> object:
        if value == "":
            return None
        return value
