from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


Feature = Annotated[float, Field(gt=0, le=10, allow_inf_nan=False, strict=True)]


class IrisFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sepal_length: Feature = Field(description="Sepal length in centimeters")
    sepal_width: Feature = Field(description="Sepal width in centimeters")
    petal_length: Feature = Field(description="Petal length in centimeters")
    petal_width: Feature = Field(description="Petal width in centimeters")

    def as_row(self) -> list[float]:
        return [self.sepal_length, self.sepal_width, self.petal_length, self.petal_width]


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instances: list[IrisFeatures] = Field(min_length=1, max_length=250)


class ModelRef(BaseModel):
    name: str
    version: str


class PredictionResponse(BaseModel):
    prediction: str
    class_index: int
    probabilities: dict[str, float]
    model: ModelRef
    latency_ms: float


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int
    model: ModelRef
    latency_ms: float
