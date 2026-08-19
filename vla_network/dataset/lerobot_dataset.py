import os
import json
from collections import OrderedDict
from collections import deque
from functools import lru_cache
from pathlib import Path
from torchvision.transforms import v2 as tvtf2
import torch
import numpy as np
import pyarrow.parquet as pq
import transforms3d as t3d
from PIL import Image
import random
import zlib

from vla_network.dataset.common import BaseDataset
from vla_network.datasample import VLASample
from vla_network.config import VLAModelConfig
from vla_network.utils.logger import log
from vla_network.utils.dataset_utils import (
    LocalLeRobotDatasetMetadata,
    LocalStreamingLeRobotDatasetShard,
)
from typing import Iterable, List, Optional


class LimitedStreamingLeRobotDataset:
    """Iterable that keeps a bounded number of active LeRobot shard readers.

    It yields items by round-robin across up to max_active_shards shards, yielding up to
    chunk_size items per shard on each turn. Shard order is shuffled for diversity.
    """

    def __init__(
        self,
        shard_paths: Iterable[Path],
        delta_timestamps: dict,
        seed: int,
        max_active_shards: int = 16,
        chunk_size: int = 1,
        tolerate_corruption: bool = True,
        frame_sample_probability: float = 1.0,
        frames_per_episode: Optional[int] = None,
    ):
        self.shard_paths: List[Path] = list(shard_paths)
        self.delta_timestamps = delta_timestamps
        self.seed = seed
        self.max_active_shards = max_active_shards
        self.chunk_size = chunk_size
        self.tolerate_corruption = tolerate_corruption
        self.frame_sample_probability = frame_sample_probability
        self.frames_per_episode = frames_per_episode

        def shard_seed(path: Path) -> int:
            path_checksum = zlib.crc32(os.fsencode(path))
            return (self.seed ^ path_checksum) & 0xFFFFFFFF

        self._shard_seed = shard_seed

        # shuffle shards using python's random with a fixed seed
        random.Random(self.seed).shuffle(self.shard_paths)

        # probe first shard metadata and fail early if none available
        self.first_meta_features = None
        for p in self.shard_paths:
            try:
                tmp = LocalStreamingLeRobotDatasetShard(
                    root=p,
                    delta_timestamps=self.delta_timestamps,
                    tolerance_s=1e-4,
                    buffer_size=1,
                    max_num_shards=1,
                    seed=self._shard_seed(p),
                    shuffle=False,
                    frame_sample_probability=self.frame_sample_probability,
                    frames_per_episode=self.frames_per_episode,
                )
                self.first_meta_features = tmp.meta.features
                break
            except Exception:
                # try next shard
                continue

        if self.first_meta_features is None:
            raise RuntimeError("Failed to read metadata from any shard in shard_paths")

    def __iter__(self):
        remaining = deque(self.shard_paths)
        active: List[tuple] = []

        def _open_shard(path: Path):
            ds = LocalStreamingLeRobotDatasetShard(
                root=path,
                delta_timestamps=self.delta_timestamps,
                tolerance_s=1e-4,
                buffer_size=1,
                max_num_shards=1,
                seed=self._shard_seed(path),
                shuffle=False,
                frame_sample_probability=self.frame_sample_probability,
                frames_per_episode=self.frames_per_episode,
            )
            return ds, iter(ds)

        # open initial active set up to the limit
        while len(active) < self.max_active_shards and remaining:
            p = remaining.popleft()
            try:
                ds, it = _open_shard(p)
                active.append((p, ds, it))
            except Exception as e:
                log.warn(f"Failed to open shard {p}: {e}")
                if self.tolerate_corruption:
                    continue
                else:
                    raise

        # round-robin chunked serving
        while active:
            # randomize active order for better mixing
            random.shuffle(active)

            next_active = []
            for path, ds, it in active:
                yielded = 0
                while yielded < self.chunk_size:
                    try:
                        item = next(it)
                        item["_dataset_root"] = str(path)
                        yielded += 1
                        yield item
                    except StopIteration:
                        # attempt to replace exhausted shard by trying next available shards
                        replaced = False
                        while remaining and not replaced:
                            p = remaining.popleft()
                            try:
                                new_ds, new_it = _open_shard(p)
                                next_active.append((p, new_ds, new_it))
                                if self.first_meta_features is None:
                                    self.first_meta_features = new_ds.meta.features
                                replaced = True
                            except Exception as e:
                                log.warn(f"Failed to open replacement shard {p}: {e}")
                                if self.tolerate_corruption:
                                    # try next remaining shard
                                    continue
                                else:
                                    raise
                        # stop yielding from this exhausted shard (it will not be kept)
                        break
                else:
                    # still has items, keep it
                    next_active.append((path, ds, it))
            active = next_active

class FrankaLeRobotConverter:
    """Converter for LeRobot data converted from the old RLDS epiclab_franka data."""

    proprio_key = "observation.state.eef_pose.eef"
    action_key = "action.eef_pose.eef"
    proprio_gripper_key = "observation.state"
    action_gripper_key = "action"
    bbox_key_template = "observation.object_bboxes.{}"
    output_embodiment = "epiclab_franka"
    max_gripper_width = 0.08

    @staticmethod
    def _xyzw_to_rotmat(quaternions):
        # LeRobot stores quaternion as x, y, z, w; transforms3d expects w, x, y, z.
        return np.array([
            t3d.quaternions.quat2mat(quaternion[[3, 0, 1, 2]])
            for quaternion in quaternions
        ])

    def _normalize_gripper_width(self, values):
        # closed=-1, open=1.
        return np.clip(2.0 * values / self.max_gripper_width - 1.0, -1.0, 1.0)

    def __call__(
        self,
        proprio,
        action,
        source_names,
        proprio_gripper,
        action_gripper,
    ):
        if proprio.shape[-1] != 7 or action.shape[-1] != 7:
            raise ValueError(
                "Expected 7-D xyz+quaternion EEF proprio/action, got "
                f"{proprio.shape[-1]} and {action.shape[-1]}"
            )
        if proprio_gripper.shape[-1] != 8 or action_gripper.shape[-1] != 8:
            raise ValueError(
                "Expected 8-D joint+gripper proprio/action, got "
                f"{proprio_gripper.shape[-1]} and {action_gripper.shape[-1]}"
            )

        proprio_gripper = self._normalize_gripper_width(proprio_gripper[:, -1])
        action_gripper = self._normalize_gripper_width(action_gripper[:, -1])
        proprio = np.concatenate([proprio, proprio_gripper[:, None]], axis=-1)
        action = np.concatenate([action, action_gripper[:, None]], axis=-1)
        return proprio, action

    def to_model_space(self, proprio, action, proprio_steps):
        # compute action_i relative to the corresponding
        # future reference proprio_i, not relative to only the current frame.
        proprio_for_input = proprio[:proprio_steps]
        ref_proprio = proprio[proprio_steps - 1:proprio_steps - 1 + len(action)]
        if len(ref_proprio) != len(action):
            raise ValueError(f"Expected {len(action)} ref proprio frames, got {len(ref_proprio)}")

        proprio_rot = self._xyzw_to_rotmat(proprio_for_input[:, 3:7])
        action_rot = self._xyzw_to_rotmat(action[:, 3:7])
        ref_rot = self._xyzw_to_rotmat(ref_proprio[:, 3:7])

        proprio_rpy = np.array([t3d.euler.mat2euler(rot) for rot in proprio_rot])
        delta_rot = np.einsum("nab,ncb->nac", action_rot, ref_rot)
        delta_rpy = np.array([t3d.euler.mat2euler(rot) for rot in delta_rot])

        proprio_for_input = np.concatenate(
            [proprio_for_input[:, :3], proprio_rpy, proprio_for_input[:, 7:8]],
            axis=-1,
        )
        action = np.concatenate(
            [action[:, :3] - ref_proprio[:, :3], delta_rpy, action[:, 7:8]],
            axis=-1,
        )
        return proprio_for_input, np.zeros(6), action

    def goal_to_model_space(self, goal):
        if goal.shape != (7,):
            raise ValueError(f"Expected one 7-D xyz+quaternion goal, got {goal.shape}")
        goal_rot = self._xyzw_to_rotmat(goal[None, 3:7])[0]
        goal_rpy = np.asarray(t3d.euler.mat2euler(goal_rot))
        return np.concatenate([goal[:3], goal_rpy])

    @staticmethod
    def augment_proprio(proprio):
        proprio = proprio.copy()
        proprio_trans = proprio[:, :3]
        proprio_rot = np.array([
            t3d.euler.euler2mat(*rpy) for rpy in proprio[:, 3:6]
        ])

        if random.random() < 0.1:
            proprio_trans = np.tile(proprio_trans[-1], (len(proprio_trans), 1))
            proprio_rot = np.tile(proprio_rot[-1], (len(proprio_rot), 1, 1))

        proprio_trans = proprio_trans + np.random.uniform(
            -0.01 / 3,
            0.01 / 3,
            size=(len(proprio_trans), 3),
        )
        proprio_rpy_noise = np.random.uniform(
            -0.05 / 3,
            0.05 / 3,
            size=(len(proprio_rot), 3),
        )
        proprio_noise = np.array([
            t3d.euler.euler2mat(*rpy) for rpy in proprio_rpy_noise
        ])
        proprio_rot = np.einsum("nab,nbc->nac", proprio_rot, proprio_noise)
        proprio_rpy = np.array([
            t3d.euler.mat2euler(rot) for rot in proprio_rot
        ])
        return np.concatenate(
            [proprio_trans, proprio_rpy, proprio[:, 6:7]],
            axis=-1,
        )


@lru_cache(maxsize=32)
def _load_episode_locations(dataset_root: str):
    """Map an episode to its data parquet file without loading image data."""
    root = Path(dataset_root)
    columns = [
        "episode_index",
        "data/chunk_index",
        "data/file_index",
    ]
    locations = {}
    episode_meta_files = sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not episode_meta_files:
        raise FileNotFoundError(f"No episode metadata parquet found under {root}")

    for meta_path in episode_meta_files:
        table = pq.read_table(meta_path, columns=columns).to_pydict()
        for episode_index, chunk_index, file_index in zip(
            table["episode_index"],
            table["data/chunk_index"],
            table["data/file_index"],
        ):
            locations[int(episode_index)] = (int(chunk_index), int(file_index))
    return locations


@lru_cache(maxsize=4096)
def _load_franka_episode_goal_data(dataset_root: str, episode_index: int):
    """Build the old RLDS next-gripper-event goal lookup for one episode."""
    locations = _load_episode_locations(dataset_root)
    if episode_index not in locations:
        raise KeyError(f"Episode {episode_index} is absent from {dataset_root}")

    chunk_index, file_index = locations[episode_index]
    data_path = (
        Path(dataset_root)
        / "data"
        / f"chunk-{chunk_index:03d}"
        / f"file-{file_index:03d}.parquet"
    )
    columns = [
        "episode_index",
        "frame_index",
        "action",
        "action.eef_pose.eef",
        "action.subtask_id",
    ]
    table = pq.read_table(
        data_path,
        columns=columns,
        filters=[("episode_index", "=", episode_index)],
    ).to_pydict()

    frames = np.asarray(table["frame_index"], dtype=np.int64)
    order = np.argsort(frames)
    frames = frames[order]
    action = np.asarray(table["action"], dtype=np.float64)[order]
    action_eef = np.asarray(table["action.eef_pose.eef"], dtype=np.float64)[order]
    subtask_ids = np.asarray(table["action.subtask_id"], dtype=np.int64)[order]
    if len(frames) == 0:
        raise ValueError(f"Episode {episode_index} in {dataset_root} is empty")

    # The converted data does not preserve the old discrete gripper_proprio
    # status. Use command and subtask transitions to approximate the original
    # gripper movement segments.
    command_width = action[:, -1]
    command_changes = np.flatnonzero(
        ~np.isclose(command_width[1:], command_width[:-1], atol=1e-6)
    ) + 1

    next_goal_indices = np.full(len(frames), len(frames) - 1, dtype=np.int64)
    last_change_index = 0
    for movement_start in command_changes:
        future_changes = np.flatnonzero(
            subtask_ids[movement_start + 1:] != subtask_ids[movement_start]
        )
        if len(future_changes) > 0:
            movement_end = movement_start + int(future_changes[0])
        else:
            movement_end = len(frames) - 1
        next_goal_indices[last_change_index:movement_end + 1] = movement_end
        last_change_index = movement_end + 1

    return frames, next_goal_indices, action_eef


def _as_str(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _as_str(value.item())
        if value.size == 1:
            return _as_str(value.reshape(-1)[0])
    if torch.is_tensor(value):
        if value.numel() == 1:
            return _as_str(value.reshape(-1)[0].item())
    return str(value)


def _as_int(value) -> int:
    if isinstance(value, np.ndarray):
        return int(value.reshape(-1)[0])
    if torch.is_tensor(value):
        return int(value.reshape(-1)[0].item())
    return int(value)


def _as_bool(value) -> bool:
    if isinstance(value, np.ndarray):
        return bool(value.reshape(-1)[0])
    if torch.is_tensor(value):
        return bool(value.reshape(-1)[0].item())
    return bool(value)


@lru_cache(maxsize=4096)
def _target_object_index(extra_info: str) -> int:
    """Find the target object's row in the fixed-size object arrays."""
    metadata = json.loads(extra_info)
    environment = metadata.get("environment_config", metadata)
    objects = environment.get("object_info", environment.get("object_infos", []))
    if isinstance(objects, dict):
        object_items = list(objects.items())
        for index, (name, obj) in enumerate(object_items):
            if name in {"target_0", "target_positive"} or obj.get("bTarget", False):
                return index
    else:
        target = environment.get("target_info", {})
        target_id = target.get("id")
        for index, obj in enumerate(objects):
            if obj.get("bTarget", False) or (
                target_id is not None and obj.get("id") == target_id
            ):
                return index
    return 0


def _extract_instruction(item: dict) -> str:
    if "task" in item:
        instruction = _as_str(item["task"])
        if instruction:
            # LeRobot task names are stored as snake_case identifiers.
            return " ".join(instruction.replace("_", " ").split())
    if "episode.extra_info" in item:
        extra_info = _as_str(item["episode.extra_info"])
        if extra_info:
            return extra_info
    return "pick up anything"


class LeRobotBaseDataset(BaseDataset):
    IMAGE_KEY_MAP = {
        'left': 'camera_front_head_left_rgb',
        'right': 'camera_front_head_right_rgb',
        'left_wrist': 'camera_left_wrist',
        'right_wrist': 'camera_right_wrist',
    }
    def __init__(self, config: VLAModelConfig, path: str, train: bool):
        super().__init__(config, path, train)
        self.dt = config.dt
        
        # TODO: the dataset class should ensure its datasamples have the same embodiment and joint order etc.
        dataset_roots = self._discover_dataset_roots(path)

        # get data_dt from the first shard's metadata
        data_fps = None
        for meta_root in dataset_roots:
            try:
                data_fps = LocalLeRobotDatasetMetadata(
                    repo_id="data",
                    root=str(meta_root),
                ).fps
                break
            except Exception as e:
                log.warn(f"Failed to load metadata from {meta_root}: {e}")
        assert data_fps is not None, f"Failed to load metadata from any shard in {path}"
        self.data_dt = 1.0 / data_fps
        self.dataset_roots = dataset_roots

        assert config.image_steps == 1, "only support single-step images for now"
        self.image_timestamps = [i * self.dt for i in range(-config.image_steps + 1, 1)]
        
        self.dataset_image_keys = [f"observation.rgbs.{self.IMAGE_KEY_MAP[k]}" for k in config.image_keys]
        
        self.image_aug_transforms = tvtf2.Compose([
            tvtf2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        ])
        self.yielded_frames = 0
        self._first_meta_features = None

    @staticmethod
    def _discover_dataset_roots(path: str) -> List[Path]:
        root = Path(path)
        if (root / "meta" / "info.json").exists():
            return [root]

        roots: List[Path] = []
        for child in root.iterdir():
            if not child.is_dir():
                continue
            if (child / "meta" / "info.json").exists():
                roots.append(child)
                continue
            # The target LeRobot path has one collection directory first, then
            # per-shard UUID directories containing meta/info.json.
            for grandchild in child.iterdir():
                if grandchild.is_dir() and (grandchild / "meta" / "info.json").exists():
                    roots.append(grandchild)
        return sorted(roots)

    def _setup_streaming_dataset(self, delta_timestamps):
        self.dataset = LimitedStreamingLeRobotDataset(
            self.dataset_roots,
            delta_timestamps,
            seed=torch.initial_seed() if not self.train else random.randint(0, int(1e9)),
            tolerate_corruption=True,
            frame_sample_probability=1.0,
            # uniformly sample ten frames per trajectory during training.
            frames_per_episode=10 if self.train else None,
        )
        # cache first meta features (must exist)
        self._first_meta_features = self.dataset.first_meta_features
        self.iterator = iter(self.dataset)

    def close(self):
        iterator = getattr(self, "iterator", None)
        if iterator is not None and hasattr(iterator, "close"):
            iterator.close()
        self.iterator = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _center_crop_and_resize(img_tensor: torch.Tensor, target_width: int, target_height: int) -> torch.Tensor:
        img_np = (img_tensor*255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        img: Image.Image = Image.fromarray(img_np)

        # Center-crop the image and resize into config.image_size x config.image_size
        w, h = img.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        right = (w + min_dim) // 2
        bottom = (h + min_dim) // 2
        img = img.crop((left, top, right, bottom))
        img = img.resize((target_width, target_height), resample=Image.LANCZOS)
        return img

    def _sample_raw_trajectory(self) -> dict[str, torch.Tensor]:
        while True:
            try:
                item = next(self.iterator)
                self.yielded_frames += 1
                break
            except StopIteration:
                log.info(f"{self.__class__.__name__} {self.path} reached StopIteration, yielded {self.yielded_frames} frames.")
                assert self.yielded_frames > 0, "Got StopIteration without yielding any frames, something might be wrong with the dataset."
                self.yielded_frames = 0
                self.iterator = iter(self.dataset)
            except Exception as e:
                log.warn(f"{self.__class__.__name__} {self.path} got exception: {repr(e)}, yielded {self.yielded_frames} frames.")
                if self.yielded_frames <= 0: raise e
                assert self.yielded_frames > 0, f"Got exception {repr(e)} without yielding any frames, something might be wrong with the dataset."
        
                self.yielded_frames = 0
                self.iterator = iter(self.dataset)
        return item

    def _load_images(self, item: dict[str, torch.Tensor]) -> OrderedDict:
        images = OrderedDict()
        for ds_key, key in zip(self.dataset_image_keys, self.config.image_keys):
            img_tensor = item[ds_key] # (C, H, W)
            img = self._center_crop_and_resize(img_tensor, self.config.image_size, self.config.image_size)
            if self.train:
                img = self.image_aug_transforms(img)
            images[key] = [img]
        return images


class LeRobotActionDataset(LeRobotBaseDataset):
    CONVERTER_MAP = {
        "epiclab_franka": FrankaLeRobotConverter(),
    }
    def __init__(self, config: VLAModelConfig, path: str, train: bool):
        super().__init__(config, path, train)
        self.is_action_dataset = True
        
        self.proprio_timestamps = [i * self.dt for i in range(-config.proprio_steps + 1, config.action_steps)]
        self.action_timestamps = [(i + 1) * self.dt - self.data_dt for i in range(config.action_steps)]

        delta_timestamps = {}
        for ds_key in self.dataset_image_keys:
            delta_timestamps[ds_key] = self.image_timestamps
        
        converter = self._default_converter()
        delta_timestamps[converter.proprio_key] = self.proprio_timestamps
        delta_timestamps[converter.action_key] = self.action_timestamps
        if hasattr(converter, "proprio_gripper_key"):
            delta_timestamps[converter.proprio_gripper_key] = self.proprio_timestamps
        if hasattr(converter, "action_gripper_key"):
            # advance gripper action_frames by one raw 0.1s step.
            delta_timestamps[converter.action_gripper_key] = [
                timestamp + self.data_dt for timestamp in self.action_timestamps
            ]
        if hasattr(converter, "bbox_key_template"):
            for source_key in set(self.IMAGE_KEY_MAP.values()):
                delta_timestamps[
                    converter.bbox_key_template.format(source_key)
                ] = self.image_timestamps
        
        self._setup_streaming_dataset(delta_timestamps)

    @classmethod
    def _default_converter(cls):
        if len(cls.CONVERTER_MAP) != 1:
            raise ValueError(
                f"{cls.__name__} has multiple converters; use a dataset-specific subclass."
            )
        return next(iter(cls.CONVERTER_MAP.values()))

    @classmethod
    def _get_converter(cls, embodiment: str):
        return cls.CONVERTER_MAP[embodiment]

    def _get_goal(self, item, converter, default_goal):
        return default_goal

    def _load_bboxes(self, item, converter, images):
        if not hasattr(converter, "bbox_key_template"):
            return None

        target_index = _target_object_index(_as_str(item["episode.extra_info"]))
        bboxs = OrderedDict()
        for model_key in self.config.image_keys:
            source_key = self.IMAGE_KEY_MAP[model_key]
            dataset_key = converter.bbox_key_template.format(source_key)
            raw_bboxes = item[dataset_key]
            if torch.is_tensor(raw_bboxes):
                raw_bboxes = raw_bboxes.cpu().numpy()
            raw_bboxes = np.asarray(raw_bboxes)
            if raw_bboxes.ndim == 2:
                raw_bboxes = raw_bboxes[None]
            if target_index >= raw_bboxes.shape[1]:
                raise IndexError(
                    f"Target object index {target_index} exceeds {dataset_key} "
                    f"shape {raw_bboxes.shape}"
                )

            bbox = raw_bboxes[-1, target_index].astype(np.float64)
            image_dataset_key = f"observation.rgbs.{source_key}"
            source_feature = self._first_meta_features[image_dataset_key]
            source_height, source_width = source_feature["shape"][:2]
            target_width, target_height = images[model_key][-1].size
            bbox *= np.array(
                [
                    target_width / source_width,
                    target_height / source_height,
                    target_width / source_width,
                    target_height / source_height,
                ]
            )
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, target_width)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, target_height)
            if (
                np.any(raw_bboxes[-1, target_index] < 0)
                or bbox[2] <= bbox[0]
                or bbox[3] <= bbox[1]
            ):
                bbox = np.zeros(4, dtype=np.float64)

            bboxs[model_key] = [bbox]
        return bboxs

    def _load_action_sample(self) -> VLASample:
        item = self._sample_raw_trajectory()
        if (
            ("episode.valid" in item and not _as_bool(item["episode.valid"]))
            or ("episode.success" in item and not _as_bool(item["episode.success"]))
        ):
            return None
        raw_embodiment = _as_str(item["episode.embodiment"])

        instruction = _extract_instruction(item)

        images = self._load_images(item)

        converter = self._get_converter(raw_embodiment)
        embodiment = getattr(converter, "output_embodiment", raw_embodiment)

        proprio = item[converter.proprio_key].cpu().numpy()
        action = item[converter.action_key].cpu().numpy()

        assert self._first_meta_features is not None
        source_feature = self._first_meta_features.get(converter.proprio_key, {})
        source_joint_names = (source_feature.get("names") or {}).get("axes")
        if hasattr(converter, "proprio_gripper_key"):
            proprio_gripper = item[converter.proprio_gripper_key].cpu().numpy()
            action_gripper = item[converter.action_gripper_key].cpu().numpy()
            proprio, action = converter(
                proprio,
                action,
                source_joint_names,
                proprio_gripper,
                action_gripper,
            )
        else:
            proprio, action = converter(proprio, action, source_joint_names)
        proprio_for_input, goal, action = converter.to_model_space(
            proprio, action, self.config.proprio_steps
        )
        if self.train and hasattr(converter, "augment_proprio"):
            proprio_for_input = converter.augment_proprio(proprio_for_input)
        goal = self._get_goal(item, converter, goal)

        bboxs = self._load_bboxes(item, converter, images)

        sample = VLASample(
            dataset_name="lerobot_franka_dataset",
            embodiment=embodiment,
            frame=item["frame_index"],
            instruction=instruction,
            images=images,
            bboxs=bboxs,
            proprio=proprio_for_input,
            goal=goal,
            action=action,
        )
        
        return sample

    def load_raw(self) -> VLASample:
        while True:
            sample = self._load_action_sample()
            if sample is not None:
                return sample


class LeRobotFrankaDataset(LeRobotActionDataset):
    """LeRobot loader for data converted from the old RLDS epiclab_franka dataset."""

    IMAGE_KEY_MAP = {
        "left": "side",
        "right": "front",
        "front": "front",
        "side": "side",
    }
    @classmethod
    def _get_converter(cls, embodiment: str):
        # Some converted shards may not preserve the exact old embodiment string.
        # This dataset type is explicitly for Franka data, so normalize it here.
        return cls.CONVERTER_MAP["epiclab_franka"]

    def _get_goal(self, item, converter, default_goal):
        dataset_root = _as_str(item["_dataset_root"])
        episode_index = _as_int(item["episode_index"])
        frame_index = _as_int(item["frame_index"])
        frames, next_goal_indices, action_eef = _load_franka_episode_goal_data(
            dataset_root,
            episode_index,
        )

        episode_offset = int(np.searchsorted(frames, frame_index))
        if episode_offset >= len(frames) or frames[episode_offset] != frame_index:
            raise KeyError(
                f"Frame {frame_index} is absent from episode {episode_index} "
                f"under {dataset_root}"
            )

        goal_index = int(next_goal_indices[episode_offset])
        goal_pose_index = max(goal_index - 1, 0)
        goal_pose = action_eef[goal_pose_index]
        return converter.goal_to_model_space(goal_pose)
