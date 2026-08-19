import argparse
from dataclasses import dataclass
from typing import Iterator, List


@dataclass
class Arg:
    terminal_name: str
    config_paths: List[str]
    arg_type: type
    default = None
    value = None

    def __post_init__(self):
        if isinstance(self.config_paths, str):
            self.config_paths = [self.config_paths]
        assert self.value is None, "value should not be set here"


@dataclass
class Args:
    args: List[Arg]

    def __iter__(self) -> Iterator[Arg]:
        return iter(self.args)


def update_args_to_config(args: Args, config):
    for arg in args:
        for path in arg.config_paths:
            path_names = path.split("/")
            c = config
            for n in path_names[:-1]:
                assert hasattr(c, n)
                c = getattr(c, n)
            assert hasattr(c, path_names[-1]), f"{path} does not exist"
            if arg.value is not None:
                setattr(c, path_names[-1], arg.value)
    return config


def get_terminal_args(args: Args) -> Args:
    parser = argparse.ArgumentParser(description="config")
    for arg in args:
        parser.add_argument(
            f"--{arg.terminal_name}", type=arg.arg_type, default=arg.default
        )
    terminal_args = parser.parse_args()
    for arg in args:
        arg.value = getattr(terminal_args, arg.terminal_name)
    return args


def get_config(args: Args, config):
    args = get_terminal_args(args)
    return update_args_to_config(args, config)
