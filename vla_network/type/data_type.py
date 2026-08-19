"""Shared data types used by Pydantic models."""

import numpy as np
from pydantic import BeforeValidator, PlainSerializer
from typing_extensions import Annotated


def _to_ndarray(value):
    return np.asarray(value)


def _serialize_ndarray(value: np.ndarray):
    return value.tolist()


NdArray = Annotated[
    np.ndarray,
    BeforeValidator(_to_ndarray),
    PlainSerializer(_serialize_ndarray, return_type=list, when_used="json"),
]
