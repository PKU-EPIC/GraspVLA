from vla_network.datasample import BaseSample
from vla_network.config import (
    VLAConfig,
    VLAModelConfig,
    MixDatasetConfig,
)
from vla_network.utils.utils import is_main_rank
from vla_network.utils.path import get_path_preprocessor
from vla_network.utils.logger import log
from vla_network.preprocessor.preprocess import DataPreprocessor

from typing import List, Optional
import random
import os
from tqdm import tqdm
from torch.utils.data import Dataset
import pickle
import tempfile
import time

from .common import BaseDataset, wait_for_file


# train: whether it is train/val
def init_dataset(config: VLAModelConfig, dataset_type: str, path: str, train: bool) -> Dataset:
    log.info(f"Initiating dataset {path}:{dataset_type}")
    kwargs = dict(config=config, path=path, train=train)
    if dataset_type == 'lerobot_franka_dataset':
        from .lerobot_dataset import LeRobotFrankaDataset
        return LeRobotFrankaDataset(**kwargs)
    raise ValueError(
        f"Unknown dataset type: {dataset_type}. "
        "This build only supports 'lerobot_franka_dataset'."
    )


class DatasetMixer(Dataset):
    datasets: List[Dataset]
    weights: List[float]

    def __init__(
        self,
        mix_config: MixDatasetConfig,
        config: VLAConfig,
        exp_name: str,
        train: bool,
        setup: bool,
        shuffle_buffer_size: int,
        reuse_preprocessor_from: Optional[str] = None,
    ):
        super().__init__()
        self.mix_config = mix_config
        self.config = config
        self.exp_name = exp_name
        self.train = train
        self.setup_flag = setup
        self.shuffle_buffer_size_orig = shuffle_buffer_size
        self.reuse_preprocessor_from = reuse_preprocessor_from

        self.specify_len = None
        self.init_pid = os.getpid()
        self._init_datasets()

    def _init_datasets(self):
        log.info(f"Initializing DatasetMixer (PID: {os.getpid()})")
        self.datasets = []
        for name, dataset_type in zip(self.mix_config.datasets, self.mix_config.datasets_type):
            dataset = init_dataset(self.config.model, dataset_type, name, self.train)
            self.datasets.append(dataset)

        self.weights = [w / sum(self.mix_config.datasets_weight) for w in self.mix_config.datasets_weight]
        self.preprocessor = DataPreprocessor(self.config)

        # evenly distribute shuffle buffer size across workers
        num_workers = self.config.train.num_workers
        self.shuffle_buffer_size = max(self.shuffle_buffer_size_orig // max(num_workers, 1), 1)

        self._buffer = []
        self.setup_preprocessor(self.train, self.setup_flag, self.exp_name, self.reuse_preprocessor_from)

    def _check_reinit(self):
        if os.getpid() != self.init_pid:
            self._init_datasets()
            self.init_pid = os.getpid()

    def specify_length(self, length: int):
        self.specify_len = length

    def __len__(self) -> int:
        return self.specify_len or sum([len(dataset) for dataset in self.datasets])

    def _get_processed_sample(self) -> BaseSample:
        dataset: BaseDataset = random.choices(self.datasets, weights=self.weights)[0]
        sample = dataset.load_raw()
        return self.preprocessor.process_input(sample, normalize=True)

    def __getitem__(self, idx: Optional[int] = None) -> BaseSample:
        self._check_reinit()
        if self.train:
            if len(self._buffer) < self.shuffle_buffer_size:
                for _ in tqdm(range(self.shuffle_buffer_size - len(self._buffer)), desc="Filling shuffle buffer"):
                    self._buffer.append(self._get_processed_sample())
            idx = random.randrange(len(self._buffer))
            sample = self._buffer[idx]
            self._buffer[idx] = self._get_processed_sample()
            return sample
        else:
            return self._get_processed_sample()

    def setup_preprocessor(self, train: bool, setup: bool, exp_name: str, reuse_preprocessor_from: Optional[str]):
        def get_data():
            action_datasets, action_datasets_weights = [], []
            for dataset, weight in zip(self.datasets, self.weights):
                if dataset.is_action_dataset:
                    action_datasets.append(dataset)
                    action_datasets_weights.append(weight)
            dataset = random.choices(action_datasets, weights=action_datasets_weights)[0]
            return dataset.load_raw()

        if reuse_preprocessor_from is not None:
            path = get_path_preprocessor(reuse_preprocessor_from)
            assert os.path.exists(path), "preprocessor not found"
        else:
            path = get_path_preprocessor(exp_name)
            os.makedirs(os.path.dirname(path), exist_ok=True)

        # FIXME: this is very tricky, easy to cause bugs when num_workers > 1
        is_original_process = os.getpid() == self.init_pid
        if train and setup and is_main_rank() and is_original_process:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    self.preprocessor = pickle.load(f)
                log.info("preprocessor loaded from existing file")
            else:
                self.preprocessor.setup(get_data)
            current_exp_path = get_path_preprocessor(exp_name)
            # Atomic write: write to temp file then rename to avoid partial reads
            dir_name = os.path.dirname(current_exp_path)
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
            with os.fdopen(fd, "wb") as f:
                pickle.dump(self.preprocessor, f)
            os.replace(tmp_path, current_exp_path)
        else:
            wait_for_file(path, dt=10)
            while True:
                try:
                    with open(path, "rb") as f:
                        self.preprocessor = pickle.load(f)
                    break
                except Exception as e:
                    log.warn(f"Failed to load preprocessor from {path}: {e}. Retrying...")
                    time.sleep(10)
            log.info("preprocessor loaded successfully")
