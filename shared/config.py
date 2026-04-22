from dataclasses import dataclass, asdict
from typing import Any
import yaml


@dataclass
class BaseConfig:
    seed: int = 42
    output_dir: str = "outputs/"
    log_every_n_steps: int = 10
    checkpoint_every_n_epochs: int = 1


def load_config(path: str, config_cls: type) -> Any:
    """Load a YAML file and instantiate config_cls with its values."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return config_cls(**data)


def save_config(config: BaseConfig, path: str) -> None:
    """Serialize config to YAML via asdict."""
    with open(path, "w") as f:
        yaml.dump(asdict(config), f, default_flow_style=False)
