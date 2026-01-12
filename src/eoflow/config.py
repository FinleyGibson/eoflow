"""
Configuration management for the eoflow package.

Provides a Config class that automatically handles loading/saving configuration
from platform-specific default locations. Creates default config if none exists.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Self, Union

from platformdirs import user_config_dir


def _get_logger():
    """Lazy logger to avoid circular import."""
    from eoflow.log_utils import get_logger

    return get_logger(__name__)


def get_default_config_dir() -> Path:
    """
    Get the platform-specific default configuration directory.

    Returns:
        Path to the config directory.

    Example locations:
        - Linux: ~/.config/eoflow/
        - macOS: ~/Library/Application Support/eoflow/
        - Windows: %APPDATA%/eoflow/
    """
    return Path(user_config_dir("eoflow", appauthor=False))


class Config:
    """
    Configuration manager for eoflow.

    Automatically loads configuration from the default config location,
    or creates a new config file with default values if none exists.

    The config file is stored as JSON for simplicity and portability.
    """

    @classmethod
    def _default_config(cls) -> Dict[str, Any]:
        """
        Get the default configuration values.

        Returns:
            Dictionary containing default configuration
        """
        return {
            "logging": {
                "name": "eoflow",
                "level": "INFO",
                "console": True,
                "colored": True,
                "log_file": None,
                "format_string": None,
                "file_level": None,
                "console_level": None,
            },
            "api": {
                "ea_base_url": "https://environment.data.gov.uk/water-quality/data/observation",
                "ea_max_limit": 2500,
                "ea_api_delay": 0.5,
                "ea_api_timeout": 30,
                "max_retries": 3,
            },
            "data": {
                "cache_dir": None,  # None means use system temp
                "cache_enabled": True,
                "cache_ttl_days": 7,
            },
            "paths": {
                "output_dir": "./output",
                "temp_dir": None,
            },
        }

    @classmethod
    def default(
        cls,
        config_file: Optional[Union[str, Path]] = None,
        auto_save: bool = True,
    ) -> Self:
        """
        Create a Config instance with default values.

        This is a convenience class method that creates a new Config instance
        and initializes it with default configuration values.

        Args:
            config_file: Path to config file. If None, uses default location.
            auto_save: If True, automatically save changes when setting values.

        Returns:
            Config instance with default configuration

        Example:
            >>> config = Config.default()
            >>> print(config.get("logging.level"))
            'INFO'
        """
        return cls(config_file=config_file, auto_save=auto_save)

    def __init__(
        self,
        config_file: Optional[Union[str, Path]] = None,
        auto_save: bool = True,
    ):
        """
        Initialize the Config object.

        Args:
            config_file: Path to config file. If None, uses default location.
            auto_save: If True, automatically save changes when setting values.

        Example:
            >>> # Use default config location
            >>> config = Config()
            >>> print(config.get("logging.level"))
            'INFO'
            >>>
            >>> # Use custom config file
            >>> config = Config("my_config.json")
        """
        self.auto_save = auto_save
        self._config: Dict[str, Any] = {}

        # Determine config file path
        if config_file is None:
            self.config_dir = get_default_config_dir()
            self.config_file = self.config_dir / "config.json"
        else:
            self.config_file = Path(config_file)
            self.config_dir = self.config_file.parent

        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Load or create config
        self.load()

    def load(self) -> None:
        """
        Load configuration from file.

        If the file doesn't exist, creates it with default values.
        If the file exists but is missing keys, merges with defaults.
        """
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    loaded_config = json.load(f)
                # Merge with defaults to ensure all keys exist
                self._config = self._merge_configs(
                    self._default_config(), loaded_config
                )
            except (json.JSONDecodeError, IOError) as e:
                _get_logger().warning(
                    f"Could not load config from {self.config_file}: {e}"
                )
                _get_logger().info("Using default configuration.")
                self._config = self._default_config()
                self.save()
        else:
            # No config file exists, use defaults and create file
            self._config = self._default_config()
            self.save()
            _get_logger().info(
                f"Created new config file at: {self.config_file}"
            )

    def save(self) -> None:
        """
        Save current configuration to file.

        Example:
            >>> config = Config()
            >>> config.set("logging.level", "DEBUG")
            >>> config.save()
        """
        try:
            with open(self.config_file, "w") as f:
                json.dump(self._config, f, indent=2)
        except IOError as e:
            _get_logger().error(
                f"Could not save config to {self.config_file}: {e}"
            )

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key: Configuration key in dot notation (e.g., "logging.level")
            default: Default value if key doesn't exist

        Returns:
            Configuration value or default

        Example:
            >>> config = Config()
            >>> level = config.get("logging.level")
            >>> delay = config.get("api.ea_api_delay")
            >>> custom = config.get("nonexistent.key", "default_value")
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.

        Args:
            key: Configuration key in dot notation (e.g., "logging.level")
            value: Value to set

        Example:
            >>> config = Config()
            >>> config.set("logging.level", "DEBUG")
            >>> config.set("api.ea_api_delay", 1.0)
        """
        keys = key.split(".")
        target = self._config

        # Navigate to the parent dict
        for k in keys[:-1]:
            if k not in target or not isinstance(target[k], dict):
                target[k] = {}
            target = target[k]

        # Set the value
        target[keys[-1]] = value

        # Auto-save if enabled
        if self.auto_save:
            self.save()

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.

        Args:
            section: Section name (e.g., "logging", "api")

        Returns:
            Dictionary containing the section's configuration

        Example:
            >>> config = Config()
            >>> logging_config = config.get_section("logging")
            >>> print(logging_config)
            {'level': 'INFO', 'console': True, ...}
        """
        return self._config.get(section, {}).copy()

    def update_section(self, section: str, values: Dict[str, Any]) -> None:
        """
        Update multiple values in a configuration section.

        Args:
            section: Section name
            values: Dictionary of values to update

        Example:
            >>> config = Config()
            >>> config.update_section("logging", {
            ...     "level": "DEBUG",
            ...     "log_file": "app.log"
            ... })
        """
        if section not in self._config:
            self._config[section] = {}

        self._config[section].update(values)

        if self.auto_save:
            self.save()

    def reset(self) -> None:
        """
        Reset configuration to defaults and save.

        Example:
            >>> config = Config()
            >>> config.set("logging.level", "DEBUG")
            >>> config.reset()  # Back to INFO
        """
        self._config = self._default_config()
        self.save()

    def to_dict(self) -> Dict[str, Any]:
        """
        Get the entire configuration as a dictionary.

        Returns:
            Copy of the configuration dictionary
        """
        return self._config.copy()

    def __getitem__(self, key: str) -> Any:
        """
        Allow dictionary-style access.

        Example:
            >>> config = Config()
            >>> level = config["logging.level"]
        """
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Allow dictionary-style assignment.

        Example:
            >>> config = Config()
            >>> config["logging.level"] = "DEBUG"
        """
        self.set(key, value)

    def __repr__(self) -> str:
        """String representation of the config."""
        return f"Config(file='{self.config_file}', config={self._config})"

    @staticmethod
    def _merge_configs(base: Dict, override: Dict) -> Dict:
        """
        Recursively merge override config into base config.

        Args:
            base: Base configuration dictionary
            override: Override configuration dictionary

        Returns:
            Merged configuration
        """
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = Config._merge_configs(result[key], value)
            else:
                result[key] = value

        return result


# Singleton instance for easy access
_config_instance: Optional[Config] = None


def get_config(reload: bool = False) -> Config:
    """
    Get the global configuration instance.

    This provides a singleton pattern for easy access to config throughout the app.

    Args:
        reload: If True, reload config from file

    Returns:
        Config instance

    Example:
        >>> from eoflow.config import get_config
        >>> config = get_config()
        >>> level = config.get("logging.level")
    """
    global _config_instance

    if _config_instance is None or reload:
        _config_instance = Config()

    return _config_instance
