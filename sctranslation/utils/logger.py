import logging
import json
from pathlib import Path


class ConfigLogger:
    def __init__(self, log_file="log.txt"):
        print(f"Logging to {log_file}")
        # Ensure the log file's directory exists
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("ConfigLogger")
        self.logger.setLevel(logging.INFO)
        # Avoid duplicate handlers in interactive environments
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file, encoding="utf-8")
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log_config(self, config):
        """Safely log config attributes, converting complex types to strings."""

        def safe_convert(val):
            if isinstance(val, (int, float, str, bool, type(None))):
                return val
            return str(val)

        # Support both objects and dicts
        if hasattr(config, "__dict__"):
            config_dict = vars(config)
        elif isinstance(config, dict):
            config_dict = config
        else:
            raise TypeError("Config must be a dict or an object with __dict__")
        safe_dict = {k: safe_convert(v) for k, v in config_dict.items()}
        config_json = json.dumps(safe_dict, indent=4, ensure_ascii=False)
        self.logger.info("Config attributes:\n%s", config_json)

    def info(self, msg, *args, **kwargs):
        """Log an info-level message."""
        self.logger.info(msg, *args, **kwargs)
