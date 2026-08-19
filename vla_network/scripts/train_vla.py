from urchin import URDF, Collision, Sphere, Geometry # type: ignore
import json
import os
from pathlib import Path
import shutil
import torch # type: ignore

from os.path import exists
from transformers import Trainer
from transformers.trainer_utils import get_last_checkpoint
import torch

from vla_network.utils.utils import is_main_rank
from vla_network.utils.git import save_code_and_git
from vla_network.utils.callback import FixTrainerStateCallback
from vla_network.utils.path import get_path_exp, get_path_config_from_root
from vla_network.utils.logger import log

from vla_network.config import get_config_from_args, VLAConfig, VLADefault, Args, Arg
from vla_network.dataset import DatasetMixer
from vla_network.model.vla import BaseVLA, VLA

# Monkey patch Trainer.save_model to handle safetensors save failures
original_save_model = Trainer.save_model

def save_model_with_fallback(self, output_dir=None, _internal_call=False):
    """Save model with fallback to PyTorch format if safetensors save fails
    """
    if output_dir is None:
        output_dir = self.args.output_dir
    
    # Try to save normally (using safetensors format)
    try:
        original_save_model(self, output_dir=output_dir, _internal_call=_internal_call)
    except RuntimeError as e:
        error_msg = str(e)
        log.warn(f"Safetensors save failed, falling back to PyTorch format: {e}")
        
        # Temporarily disable safetensors, use PyTorch format
        original_save_safetensors = self.args.save_safetensors
        self.args.save_safetensors = False
        
        # Retry save with PyTorch format
        original_save_model(self, output_dir=output_dir, _internal_call=_internal_call)
        log.info("Successfully saved model in PyTorch format after fallback")

Trainer.save_model = save_model_with_fallback


def rm_r(path: str) -> None:
    target = Path(path)
    if target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        target.unlink()


args = Args(
    [
        Arg("exp_name", "train/exp_name", str),
        Arg("global_bs", "train/global_batch_size", int),
        Arg("device_bs", "train/device_batch_size", int),
        Arg("llm_mode", "model/model_specific/llm_mode", str),
        Arg("deepspeed", "train/deepspeed", str),
        Arg("fsdp", "train/fsdp", str),
        Arg("save_step", "train/save_step", int),
        Arg("max_steps", "train/max_steps", int),
        Arg("train_datasets", "train/train_datasets", str),
        Arg("val_datasets", "train/val_datasets", str),
        Arg("num_workers", "train/num_workers", int),
        Arg("profiler", "train/profiler", int),
        Arg("count_num", "train/normalizer_num_samples", int),
        Arg("save_total_limit", "train/save_total_limit", int),
        Arg("grad_ckpt", "train/gradient_checkpointing", int),
        Arg("pred", "model/model_specific/pred", str),
        Arg("attn_impl", "model/model_specific/llm/attn_implementation", str),
        Arg("action_expert", "model/model_specific/action_expert", int),
        Arg("allow_resume", "train/allow_resume", int),
        Arg("reuse_preprocessor", "train/reuse_preprocessor", int),
        Arg("ckpt", "model/ckpt", str),
        Arg("backbone_2d", "model/model_specific/backbone_2d/name", str),
        Arg("image_keys", "model/image_keys", str),
        Arg("dt", "model/dt", float),
        Arg("proprio_steps", "model/proprio_steps", int),
        Arg("action_steps", "model/action_steps", int),
        Arg("max_proprio_dim", "model/max_proprio_dim", int),
        Arg("max_action_dim", "model/max_action_dim", int),
        Arg("max_goal_dim", "model/max_goal_dim", int),
        Arg("shuffle_buffer_size", "train/shuffle_buffer_size", int),
        Arg("eval_step", "train/eval_step", int),
    ]
)


def save_cfg_code(config: VLAConfig):
    # Check existence of the output directory
    exp_root = get_path_exp(config.train.exp_name)
    if is_main_rank():
        if exists(exp_root):
            if config.train.exp_name == "debug":
                rm_r(config.train.args.output_dir)
            elif config.train.allow_resume:
                if get_last_checkpoint(config.train.args.output_dir) is not None:
                    log.info(f"Will resume experiment in {exp_root}")
                else:
                    rm_r(config.train.args.output_dir)
            else:
                if get_last_checkpoint(config.train.args.output_dir) is not None:
                    raise FileExistsError(
                        f"Output directory {config.train.args.output_dir} already exists and contains a checkpoint, but allow_resume is False."
                    )
                else:
                    rm_r(config.train.args.output_dir)

        # Save the config to JSON
        config_path = get_path_config_from_root(exp_root)
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as file:
            json.dump(
                config.model_dump(),
                file,
                indent=4,
                default=lambda value: value.tolist(),
            )

        # Save the code
        save_code_and_git(exp_root)


def main():
    if "DEBUG_PORT" in os.environ and os.environ["RANK"] == "0":
        import debugpy # type: ignore
        debugpy.listen(int(os.environ['DEBUG_PORT']))
        print(f"debugger can attach at {os.environ['DEBUG_PORT']}")

    config: VLAConfig = get_config_from_args(args, VLADefault)
    save_cfg_code(config)

    model: BaseVLA = VLA(config.model)
    model.requires_grad_(True)

    checkpoint_dir = None
    if config.train.allow_resume:
        checkpoint_dir = get_last_checkpoint(config.train.args.output_dir)
        if checkpoint_dir is not None:
            log.info(f"Checkpoint detected, resuming training from {checkpoint_dir}")

    if checkpoint_dir is not None:
        reuse_preprocessor_from = os.path.dirname(checkpoint_dir)
    elif config.model.ckpt is not None and config.train.reuse_preprocessor:
        reuse_preprocessor_from = os.path.dirname(os.path.dirname(config.model.ckpt))
    else:
        reuse_preprocessor_from = None

    train_dataset = DatasetMixer(
        config.train.train_datasets,
        config,
        config.train.exp_name,
        train=True,
        setup=True,
        reuse_preprocessor_from=reuse_preprocessor_from,
        shuffle_buffer_size=config.train.shuffle_buffer_size,
    )
    if config.train.val_datasets is None:
        config.train.val_datasets = config.train.train_datasets
    val_dataset = DatasetMixer(
        config.train.val_datasets,
        config,
        config.train.exp_name,
        train=False,
        setup=False,
        reuse_preprocessor_from=reuse_preprocessor_from,
        shuffle_buffer_size=0,
    )
    if config.train.full_bf16:
        model = model.to(torch.bfloat16)
    val_dataset.specify_length(config.train.eval_each * config.train.global_batch_size)
    trainer = Trainer(
        model=model,
        args=config.train.args,
        train_dataset=train_dataset,
        data_collator=model.collate,
        eval_dataset=val_dataset,
        callbacks=[FixTrainerStateCallback(config.train.args)],
        # callbacks=[ProfileCallback()],
    )

    log.info("Start training")
    trainer.train(resume_from_checkpoint=checkpoint_dir)


if __name__ == "__main__":
    main()
