from vla_network.datasample import BaseSample

import os
import time

from vla_network.config import (
    VLAModelConfig
)

from abc import ABC, abstractmethod


def wait_for_file(path: str, dt: float = 1.0) -> None:
    """Wait for a writer to create and close a shared file."""
    while not os.path.exists(path):
        time.sleep(dt)
    time.sleep(dt)


class BaseDataset(ABC):
    config: VLAModelConfig

    def __init__(self, config: VLAModelConfig, path: str, train: bool):
        self.config = config
        self.path = path
        self.train = train

    def __len__(self) -> int:
        """The number of samples of this dataset.

        If it is hard to determine, use an estimated number or the default value.
        """
        return int(1e7)

    @abstractmethod
    def load_raw(self) -> BaseSample:
        raise NotImplementedError()
