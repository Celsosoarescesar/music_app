from pydantic import BaseModel, Field, model_validator

from app.backends.base import EditMode


class Text2MusicRequest(BaseModel):
    tags: str = Field(min_length=1, max_length=1000)
    lyrics: str = Field(default="", max_length=5000)
    duration: float = Field(gt=0, le=600)
    seed: int | None = None
    steps: int = Field(default=27, ge=1, le=200)
    guidance_scale: float = Field(default=7.5, ge=0.0, le=20.0)


class RetakeRequest(Text2MusicRequest):
    variance: float = Field(default=0.5, ge=0.0, le=1.0)


class RepaintParams(BaseModel):
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    tags: str = Field(min_length=1, max_length=1000)
    lyrics: str = Field(default="", max_length=5000)

    @model_validator(mode="after")
    def check_time_range(self) -> "RepaintParams":
        if self.start_time >= self.end_time:
            raise ValueError("start_time must be less than end_time")
        return self


class EditParams(BaseModel):
    tags: str = Field(min_length=1, max_length=1000)
    lyrics: str = Field(default="", max_length=5000)
    mode: EditMode = "only_lyrics"


class ExtendParams(BaseModel):
    left_extend_seconds: float = Field(default=0.0, ge=0.0, le=300.0)
    right_extend_seconds: float = Field(default=0.0, ge=0.0, le=300.0)
    tags: str = Field(default="", max_length=1000)
    lyrics: str = Field(default="", max_length=5000)

    @model_validator(mode="after")
    def check_at_least_one_side(self) -> "ExtendParams":
        if self.left_extend_seconds <= 0 and self.right_extend_seconds <= 0:
            raise ValueError(
                "at least one of left_extend_seconds/right_extend_seconds must be > 0"
            )
        return self


class Audio2AudioParams(BaseModel):
    tags: str = Field(min_length=1, max_length=1000)
    lyrics: str = Field(default="", max_length=5000)
