"""Local-only LeRobot dataset helpers used by the VLA dataloaders.

This module replaces the former private dataset helper package. It builds on
the public LeRobot 0.4.2 API and carries the local-data fixes needed for
LeRobot v3 shards.
"""

from collections.abc import Generator, Iterator
import contextlib
import os
from pathlib import Path
from unittest.mock import patch

import datasets
import av
import numpy as np
import pyarrow.parquet as pq
import torch

from lerobot.datasets.lerobot_dataset import (
    CODEBASE_VERSION,
    LeRobotDatasetMetadata,
)
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.datasets.utils import Backtrackable, item_to_torch, safe_shard
from lerobot.datasets.video_utils import VideoDecoderCache
from lerobot.utils.constants import HF_LEROBOT_HOME


def _is_depth_key(key: str) -> bool:
    return key.startswith("observation.depths.")


def _decode_packed_depth_tensor(packed: torch.Tensor) -> torch.Tensor:
    """Decode packed RGB uint16-millimeter depth into float meters."""
    if not isinstance(packed, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(packed)}")

    def decode_hwc(value: torch.Tensor) -> torch.Tensor:
        packed_u8 = (value * 255.0).round().clamp(0, 255).to(torch.uint8)
        low = packed_u8[..., 0].to(torch.int32)
        high = packed_u8[..., 1].to(torch.int32)
        return (low + (high << 8)).to(torch.float32) * 0.001

    if packed.ndim == 3:
        if packed.shape[0] != 3:
            raise ValueError(f"Expected CHW packed depth, got {packed.shape}")
        return decode_hwc(packed.permute(1, 2, 0)).unsqueeze(0)

    if packed.ndim == 4:
        if packed.shape[1] != 3:
            raise ValueError(f"Expected NCHW packed depth, got {packed.shape}")
        return decode_hwc(packed.permute(0, 2, 3, 1)).unsqueeze(1)

    raise ValueError(f"Expected CHW or NCHW packed depth, got {packed.shape}")


def _decode_video_frames_pyav(
    video_path: Path,
    timestamps: list[float],
    tolerance_s: float,
) -> torch.Tensor:
    """Decode nearest RGB frames with PyAV without TorchVision VideoReader."""
    if not timestamps:
        raise ValueError("At least one timestamp is required")

    target = np.asarray(timestamps, dtype=np.float64)
    loaded_timestamps = []
    loaded_frames = []

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
        seek_timestamp = max(float(target.min()) - 1.0, 0.0)
        container.seek(
            int(seek_timestamp / time_base),
            stream=stream,
            backward=True,
            any_frame=False,
        )

        stop_timestamp = float(target.max()) + max(tolerance_s, time_base)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * stream.time_base)
            loaded_timestamps.append(timestamp)
            loaded_frames.append(frame.to_ndarray(format="rgb24"))
            if timestamp >= stop_timestamp:
                break

    if not loaded_frames:
        raise RuntimeError(f"No video frames decoded from {video_path}")

    loaded = np.asarray(loaded_timestamps)
    nearest_indices = np.abs(target[:, None] - loaded[None, :]).argmin(axis=1)
    errors = np.abs(target - loaded[nearest_indices])
    if np.any(errors >= tolerance_s):
        raise RuntimeError(
            f"Video timestamps exceed tolerance {tolerance_s}: {errors}; "
            f"path={video_path}"
        )

    frames = np.stack([loaded_frames[index] for index in nearest_indices])
    return torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255.0)


class _LocalParquetDataset:
    """Small synchronous iterable compatible with LeRobot's shard helper."""

    def __init__(self, files: list[Path]):
        if not files:
            raise FileNotFoundError("No parquet files found for local LeRobot shard")
        self.files = files

    @property
    def num_shards(self) -> int:
        return len(self.files)

    def shard(self, num_shards: int, index: int):
        return _LocalParquetDataset(self.files[index::num_shards])

    def __iter__(self):
        for path in self.files:
            parquet_file = pq.ParquetFile(path)
            for row_group_index in range(parquet_file.num_row_groups):
                rows = parquet_file.read_row_group(row_group_index).to_pylist()
                yield from rows


class LocalLeRobotDatasetMetadata(LeRobotDatasetMetadata):
    """Read LeRobot metadata strictly from a local dataset root."""

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
        metadata_buffer_size: int = 10,
    ):
        del force_cache_sync
        self.repo_id = repo_id
        self.revision = revision if revision else CODEBASE_VERSION
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
        self.writer = None
        self.latest_episode = None
        self.metadata_buffer: list[dict] = []
        self.metadata_buffer_size = metadata_buffer_size

        try:
            self.load_metadata()
        except Exception as error:
            raise FileNotFoundError(
                f"Dataset metadata is missing or incomplete under {self.root}: {error}"
            ) from error

    def pull_from_repo(self, **kwargs):
        del kwargs
        raise FileNotFoundError(
            f"Dataset metadata is not available locally under {self.root}; "
            "network downloads are disabled."
        )


def _patched_load_dataset(path_or_repo, *args, **kwargs):
    """Use the parquet builder directly for local streaming shards."""
    original_load_dataset = datasets.load_dataset
    if os.path.isdir(str(path_or_repo)) and "data_files" in kwargs:
        return original_load_dataset(
            "parquet",
            *args,
            data_files=os.path.join(str(path_or_repo), kwargs["data_files"]),
            split=kwargs.get("split"),
            streaming=kwargs.get("streaming", True),
        )
    return original_load_dataset(path_or_repo, *args, **kwargs)


@contextlib.contextmanager
def local_only_lerobot():
    """Force LeRobot metadata and streaming reads to remain local."""
    old_offline = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        with (
            patch(
                "lerobot.datasets.lerobot_dataset.LeRobotDatasetMetadata",
                LocalLeRobotDatasetMetadata,
            ),
            patch(
                "lerobot.datasets.lerobot_dataset.snapshot_download",
                side_effect=FileNotFoundError,
            ),
            patch(
                "lerobot.datasets.lerobot_dataset.get_safe_version",
                side_effect=lambda repo_id, version: version,
            ),
            patch(
                "lerobot.datasets.streaming_dataset.LeRobotDatasetMetadata",
                LocalLeRobotDatasetMetadata,
            ),
            patch(
                "lerobot.datasets.streaming_dataset.load_dataset",
                side_effect=_patched_load_dataset,
            ),
            patch("huggingface_hub.snapshot_download", side_effect=FileNotFoundError),
            patch("huggingface_hub.constants.HF_HUB_OFFLINE", True),
            patch("datasets.config.HF_HUB_OFFLINE", True),
            patch(
                "lerobot.datasets.utils.get_safe_version",
                side_effect=lambda repo_id, version: version,
            ),
        ):
            yield
    finally:
        if old_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = old_offline


class LocalStreamingLeRobotDatasetShard(StreamingLeRobotDataset):
    """Public-LeRobot streaming reader with local timestamp fixes."""

    def __init__(
        self,
        root: str | Path | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        buffer_size: int = 1000,
        max_num_shards: int = 16,
        seed: int = 42,
        rng: np.random.Generator | None = None,
        shuffle: bool = True,
        frame_sample_probability: float = 1.0,
        frames_per_episode: int | None = None,
    ):
        if not 0.0 < frame_sample_probability <= 1.0:
            raise ValueError(
                "frame_sample_probability must be in (0, 1], got "
                f"{frame_sample_probability}"
            )
        if frames_per_episode is not None and frames_per_episode <= 0:
            raise ValueError(
                f"frames_per_episode must be positive, got {frames_per_episode}"
            )
        self.frame_sample_probability = frame_sample_probability
        self.frames_per_episode = frames_per_episode
        self.frame_sample_rng = np.random.default_rng(seed)
        self._sampled_frames_by_episode: dict[int, dict[int, int]] = {}
        self._pending_sampled_items: list[dict] = []
        with local_only_lerobot():
            super().__init__(
                repo_id="data",
                root=root,
                delta_timestamps=delta_timestamps,
                tolerance_s=tolerance_s,
                force_cache_sync=False,
                streaming=True,
                buffer_size=buffer_size,
                max_num_shards=max_num_shards,
                seed=seed,
                rng=rng,
                shuffle=shuffle,
            )
        parquet_files = sorted(self.root.glob("data/*/*.parquet"))
        self.hf_dataset = _LocalParquetDataset(parquet_files)
        self.num_shards = min(self.hf_dataset.num_shards, max_num_shards)

    def _next_sampled_raw_item(self, dataset_iterator: Backtrackable) -> dict:
        while True:
            if self._pending_sampled_items:
                return self._pending_sampled_items.pop()
            raw_item = next(dataset_iterator)
            if self.frames_per_episode is not None:
                episode_index = int(raw_item["episode_index"])
                sampled_frames = self._sampled_frames_by_episode.get(episode_index)
                if sampled_frames is None:
                    episode_length = int(self.meta.episodes[episode_index]["length"])
                    # trim the first and last trajectory frames
                    # before uniformly drawing subsample_num=10 indices.
                    eligible_frames = np.arange(1, max(episode_length - 1, 1))
                    if len(eligible_frames) > self.frames_per_episode:
                        selected_frames = self.frame_sample_rng.choice(
                            eligible_frames,
                            size=self.frames_per_episode,
                            replace=True,
                        )
                    else:
                        selected_frames = eligible_frames
                    frame_values, frame_counts = np.unique(
                        selected_frames, return_counts=True
                    )
                    sampled_frames = dict(
                        zip(frame_values.tolist(), frame_counts.tolist())
                    )
                    self._sampled_frames_by_episode[episode_index] = sampled_frames

                repeat_count = sampled_frames.get(int(raw_item["frame_index"]), 0)
                if repeat_count > 0:
                    self._pending_sampled_items.extend(
                        raw_item.copy() for _ in range(repeat_count - 1)
                    )
                    return raw_item
                continue
            if (
                self.frame_sample_probability >= 1.0
                or self.frame_sample_rng.random() < self.frame_sample_probability
            ):
                return raw_item

    def make_frame(self, dataset_iterator: Backtrackable) -> Generator:
        """Build one sample using episode-relative timestamps for video queries."""
        # Sampling happens before tensor conversion, delta-window collection,
        # goal reconstruction, and—most importantly—video decoding.
        item = item_to_torch(self._next_sampled_raw_item(dataset_iterator))
        updates = []
        episode_index = item["episode_index"]
        current_timestamp = item["timestamp"]
        episode_boundaries = {
            key: (
                self.meta.episodes[episode_index][f"videos/{key}/from_timestamp"],
                self.meta.episodes[episode_index][f"videos/{key}/to_timestamp"],
            )
            for key in self.meta.video_keys
        }

        if self.delta_indices is not None:
            query_result, padding = self._get_delta_frames(dataset_iterator, item)
            updates.extend([query_result, padding])

        if self.meta.video_keys:
            original_timestamps = self._make_timestamps_from_indices(
                current_timestamp,
                self.delta_indices,
            )
            query_timestamps = self._get_query_timestamps(
                current_timestamp,
                self.delta_indices,
                episode_boundaries,
            )
            video_frames = self._query_videos(query_timestamps, episode_index)

            if self.image_transforms is not None:
                for camera_key in self.meta.camera_keys:
                    video_frames[camera_key] = self.image_transforms(
                        video_frames[camera_key]
                    )
            updates.append(video_frames)

            if self.delta_indices is not None:
                for key, timestamps in original_timestamps.items():
                    if key in self.meta.video_keys:
                        from_timestamp = episode_boundaries[key][0]
                        original_timestamps[key] = [
                            timestamp + from_timestamp for timestamp in timestamps
                        ]
                updates.append(
                    self._get_video_frame_padding_mask(
                        video_frames,
                        query_timestamps,
                        original_timestamps,
                    )
                )

        result = item.copy()
        for update in updates:
            result.update(update)
        result["task"] = self.meta.tasks.iloc[item["task_index"]].name
        yield result

    @staticmethod
    def _infinite_generator_over_elements(
        rng: np.random.Generator,
        elements: list[int],
    ) -> Iterator[int]:
        while True:
            yield rng.choice(elements)

    def _iter_frames(self) -> Iterator[dict[str, torch.Tensor]]:
        """Mirror the internal fork's fix: do not hide genuine RuntimeErrors."""
        if self.video_decoder_cache is None:
            self.video_decoder_cache = VideoDecoderCache()

        rng = np.random.default_rng(self.seed) if not self.shuffle else self.rng
        buffer_indices = self._iter_random_indices(rng, self.buffer_size)
        datasets_by_shard = {
            index: self._make_backtrackable_dataset(
                safe_shard(self.hf_dataset, index, self.num_shards)
            )
            for index in range(self.num_shards)
        }
        frames_buffer = []

        try:
            while available_shards := list(datasets_by_shard):
                shard_key = next(
                    self._infinite_generator_over_elements(rng, available_shards)
                )
                backtrackable = datasets_by_shard[shard_key]
                try:
                    frame = next(self.make_frame(backtrackable))
                    if len(frames_buffer) == self.buffer_size:
                        buffer_index = next(buffer_indices)
                        yield frames_buffer[buffer_index]
                        frames_buffer[buffer_index] = frame
                    else:
                        frames_buffer.append(frame)
                except (StopIteration, RuntimeError) as error:
                    if (
                        isinstance(error, RuntimeError)
                        and "generator raised StopIteration" not in str(error)
                    ):
                        raise
                    del datasets_by_shard[shard_key]

            rng.shuffle(frames_buffer)
            yield from frames_buffer
        finally:
            for backtrackable in datasets_by_shard.values():
                source = getattr(backtrackable, "_source", None)
                close = getattr(source, "close", None)
                if close is not None:
                    try:
                        close()
                    except BaseException:
                        pass
            datasets_by_shard.clear()

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
        episode_boundaries_ts: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        timestamps_by_key = self._make_timestamps_from_indices(
            current_ts,
            query_indices,
        )
        for key in self.meta.video_keys:
            from_timestamp, to_timestamp = episode_boundaries_ts[key]
            if query_indices is not None and key in query_indices:
                timestamps = [
                    timestamp + from_timestamp
                    for timestamp in timestamps_by_key[key]
                ]
                query_timestamps[key] = torch.clamp(
                    torch.tensor(timestamps),
                    from_timestamp,
                    to_timestamp,
                ).tolist()
            else:
                query_timestamps[key] = [current_ts + from_timestamp]
        return query_timestamps

    def _query_videos(
        self,
        query_timestamps: dict[str, list[float]],
        episode_index: int,
    ) -> dict[str, torch.Tensor]:
        """Decode with public PyAV, avoiding TorchCodec/TorchVision ABI issues."""
        item = {}
        for video_key, timestamps in query_timestamps.items():
            root = (
                self.meta.url_root
                if self.streaming and not self.streaming_from_local
                else self.root
            )
            video_path = root / self.meta.get_video_file_path(
                episode_index,
                video_key,
            )
            frames = _decode_video_frames_pyav(
                video_path,
                timestamps,
                self.tolerance_s,
            )
            item[video_key] = (
                frames.squeeze(0) if len(timestamps) == 1 else frames
            )
        return item

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for item in self._iter_frames():
            for key in list(item):
                feature = self.meta.features.get(key)
                if (
                    feature
                    and feature.get("dtype") == "video"
                    and feature.get("video_encoding") == "depth_lossless"
                ):
                    item[key] = _decode_packed_depth_tensor(item[key])
            yield item
