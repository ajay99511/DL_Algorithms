import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JSONLogger:
    """Appends one JSON object per line to a log file."""

    def __init__(self, log_path: str) -> None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path

    def log(self, entry: dict[str, Any]) -> None:
        """Append a single JSON object as one line to the log file."""
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def log_config(self, config: Any) -> None:
        """Write the full config as the first log entry with metadata."""
        import torch
        import numpy

        try:
            import transformers
            transformers_version = transformers.__version__
        except ImportError:
            transformers_version = "not installed"

        # Support dataclasses, objects with __dict__, or plain dicts
        from dataclasses import asdict, is_dataclass
        if is_dataclass(config):
            config_dict = asdict(config)
        elif isinstance(config, dict):
            config_dict = config
        else:
            config_dict = vars(config)

        entry = {
            "type": "config",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": config_dict,
            "library_versions": {
                "torch": torch.__version__,
                "transformers": transformers_version,
                "numpy": numpy.__version__,
            },
        }
        self.log(entry)
