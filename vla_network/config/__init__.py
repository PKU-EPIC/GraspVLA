from .args import Arg, Args, get_config  # type: ignore
from .define import Backbone2DConfig, LLMConfig, VLAModelConfig, VLATrainConfig, VLAConfig, MixDatasetConfig, ActionExpertConfig, FlowMatchingConfig, AssembledVLAModelConfig, STATE_REP, ACTION_REP  # type: ignore


VLAModelDefault = VLAModelConfig(
    model_type="AssembledVLA",
    model_specific=AssembledVLAModelConfig(
        backbone_2d=Backbone2DConfig(name="dinosiglip", image_size=224),
        llm=LLMConfig(
            name="internlm/internlm2-1_8b",
            attn_implementation="sdpa",
        ),
        pred="cot_flow_matching",
        action_expert=1,
        action_expert_cfg=ActionExpertConfig(
            hidden_size_scale=2,
            intermediate_size_scale=4,
        ),
        flow_matching_cfg=FlowMatchingConfig(
            beta_alpha=1.5,
            beta_beta=1.0,
            time_min=0.001,
            time_max=1.0,
        ),
        backbone_2d_mode="freeze",
        projector_mode="train",
        llm_mode="train",
        action_token_num=256,
    ),
    ckpt=None,
    max_proprio_dim=7,
    max_action_dim=7,
    max_goal_dim=6,

    proprio_rep=STATE_REP.EEF,
    goal_rep=STATE_REP.EEF,
    action_rep=ACTION_REP.REL_EEF,
    dt=0.3,
    proprio_steps=2,
    action_steps=4,
    image_steps=1,
    image_keys=["left", "right"],
    image_size=224,
)


VLATrainDefault = VLATrainConfig(
    exp_name="debug",
    max_steps=25000,
    global_batch_size=512,
    device_batch_size=64,
    lr=1.6e-4,
    lr_scheduler_type="constant",
    weight_decay=0.0,
    max_grad_norm=1.0,
    warmup_ratio=0.0,
    log_step=10,
    save_step=2500,
    save_total_limit=1000000000,
    eval_step=2500,
    eval_each=25,
    num_workers=64,
    shuffle_buffer_size=40000,
    deepspeed=None,
    fsdp="full_shard",
    profiler=False,
    bf16=True,
    full_bf16=True,
    gradient_checkpointing=False,
    normalizer_ratio_limit=0.01,
    normalizer_num_samples=10000,
)

VLADefault = VLAConfig(
    model=VLAModelDefault,
    train=VLATrainDefault,
)

def get_config_from_args(args: Args, default: VLAConfig) -> VLAConfig:
    config: VLAConfig = get_config(args, default)
    # Normalize CLI-provided image_keys (allow comma-separated string)
    if isinstance(config.model.image_keys, str):
        config.model.image_keys = [k.strip() for k in config.model.image_keys.split(",") if k.strip()]
    config.train.setup()
    return config
