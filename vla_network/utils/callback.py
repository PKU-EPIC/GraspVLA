from transformers import TrainerCallback
from transformers.trainer import TrainingArguments

from vla_network.utils.logger import log


class FixTrainerStateCallback(TrainerCallback):
    """
    Huggingface trainer ignores training arguments on resume. Fix it with this class.
    """
    def __init__(self, args: TrainingArguments):
        self.args = args
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        to_fix = ["save_steps", "max_steps", "logging_steps", "eval_steps"]
        for name in to_fix:
            args_val = getattr(self.args, name)
            state_val = getattr(state, name)
            if state_val != args_val:
                log.info(f"fix trainer.state.{name} from {state_val} to {args_val}")
                setattr(state, name, args_val)
        return control


# class ProfileCallback(TrainerCallback):
#     def on_step_begin(self, args, state, control, logs=None, **kwargs):
#         profiler.clear()
#         profiler.enter("step")
#         profiler.enter("load_data")

#     def on_pre_optimizer_step(self, args, state, control, **kwargs):
#         profiler.exit("model_backward")
#         profiler.enter("optimizer")

#     def on_optimizer_step(self, args, state, control, **kwargs):
#         profiler.exit("optimizer")
#         profiler.enter("post_optimizer")

#     def on_step_end(self, args, state, control, logs=None, **kwargs):
#         profiler.exit("post_optimizer")
#         profiler.exit("step")
#         profiler.show()
