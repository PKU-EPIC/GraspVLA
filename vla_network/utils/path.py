"""Storage, experiment, checkpoint, and model asset paths."""

import os
from pathlib import Path
from typing import Optional


CONFIG_JSON = "config.json"
MODEL_FILE = "model.safetensors"
PRETRAINED = "pretrained"
PREPROCESSOR_NPZ = "preprocessor.npz"


def get_storage_path() -> Path:
    configured = os.getenv("VLA_STORAGE_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "storage"


def get_dir_ckpt() -> str:
    return str(get_storage_path() / "ckpt")


def get_dir_exp() -> str:
    return str(Path(get_dir_ckpt()) / "exp")


def get_path_exp(exp: str) -> str:
    return str(Path(get_dir_exp()) / exp)


def get_path_ckpt(
    exp: Optional[str] = None,
    it: Optional[int] = None,
    exp_path: Optional[str] = None,
) -> str:
    if exp is not None:
        exp_path = get_path_exp(exp)
    if exp_path is None:
        raise ValueError("Either exp or exp_path must be provided")
    return str(Path(exp_path) / f"checkpoint-{it}" / MODEL_FILE)


def get_path_config_from_root(root: str) -> str:
    return str(Path(root) / CONFIG_JSON)


def get_path_exp_config(
    exp: Optional[str] = None,
    exp_path: Optional[str] = None,
) -> str:
    if exp is not None:
        exp_path = get_path_exp(exp)
    if exp_path is None:
        raise ValueError("Either exp or exp_path must be provided")
    return get_path_config_from_root(exp_path)


def get_path_exp_from_ckpt(ckpt_path: str) -> str:
    return str(Path(ckpt_path).parent.parent)


def get_dir_pretrained() -> str:
    return str(Path(get_dir_ckpt()) / PRETRAINED)


def get_path_pretrained(path: str) -> str:
    return str(Path(get_dir_pretrained()) / path)


def get_path_preprocessor(
    exp: Optional[str] = None,
    exp_path: Optional[str] = None,
) -> str:
    if exp is not None:
        exp_path = get_path_exp(exp)
    if exp_path is None:
        raise ValueError("Either exp or exp_path must be provided")
    return str(Path(exp_path) / PREPROCESSOR_NPZ)