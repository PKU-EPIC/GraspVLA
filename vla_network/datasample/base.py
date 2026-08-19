from pydantic import BaseModel, ConfigDict


class BaseSample(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset_name: str
    """The name of the dataset this sample was from, to aid debugging."""
