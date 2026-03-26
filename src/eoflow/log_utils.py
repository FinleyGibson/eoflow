"""
Logging utilities for the eoflow package.

Provides convenient logging setup with support for console and file output,
colored console output, and flexible configuration.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union

# Default format strings
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DETAILED_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - "
    "%(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
)
SIMPLE_FORMAT = "%(levelname)s - %(name)s - %(message)s"

# ANSI color codes for console output
COLORS = {
    "DEBUG": "\033[36m",  # Cyan
    "INFO": "\033[32m",  # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[35m",  # Magenta
    "RESET": "\033[0m",  # Reset
}


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to console output."""

    def format(self, record):
        original_levelname = record.levelname
        if original_levelname in COLORS:
            record.levelname = f"{COLORS[original_levelname]}{original_levelname}{COLORS['RESET']}"

        formatted = super().format(record)
        record.levelname = original_levelname  # restore
        return formatted


def setup_logging(
    name: Optional[str] = None,
    level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    console: bool = True,
    colored: bool = True,
    format_string: Optional[str] = None,
    file_level: Optional[Union[str, int]] = None,
    console_level: Optional[Union[str, int]] = None,
) -> logging.Logger:
    """
    Set up logging with console and/or file output.

    Args:
        name: Logger name. If None, returns root logger.
        level: Default logging level (applies to logger and handlers if not specified).
        log_file: Path to log file. If provided, file handler will be added.
        console: Whether to add console handler.
        colored: Whether to use colored output for console (ignored for file).
        format_string: Custom format string. If None, uses DEFAULT_FORMAT.
        file_level: Logging level for file handler. If None, uses `level`.
        console_level: Logging level for console handler. If None, uses `level`.

    Returns:
        Configured logger instance.

    Example:
        >>> # Basic setup with console output
        >>> logger = setup_logging("myapp")
        >>> logger.info("Application started")
        >>>
        >>> # Setup with file output and detailed formatting
        >>> logger = setup_logging(
        ...     "myapp",
        ...     level="DEBUG",
        ...     log_file="app.log",
        ...     format_string=DETAILED_FORMAT
        ... )
    """
    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Determine format string
    fmt = format_string or DEFAULT_FORMAT

    # Add console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level or level)

        if colored and sys.stdout.isatty():
            console_formatter = ColoredFormatter(fmt)
        else:
            console_formatter = logging.Formatter(fmt)

        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # Add file handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(file_level or level)

        # Never use colors in file output
        file_formatter = logging.Formatter(fmt)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to avoid duplicate logs
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    This is a convenience function that returns a logger without configuring it.
    Use setup_logging() for initial configuration.

    Args:
        name: Logger name (typically __name__ of the calling module).

    Returns:
        Logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    return logging.getLogger(name)


def set_level(logger: Union[logging.Logger, str], level: Union[str, int]) -> None:
    """
    Set the logging level for a logger and all its handlers.

    Args:
        logger: Logger instance or logger name.
        level: New logging level (e.g., "DEBUG", logging.DEBUG, etc.).

    Example:
        >>> logger = get_logger("myapp")
        >>> set_level(logger, "DEBUG")
        >>> # Or by name
        >>> set_level("myapp", logging.WARNING)
    """
    if isinstance(logger, str):
        logger = logging.getLogger(logger)

    if isinstance(level, str):
        level = getattr(logging, level.upper())

    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def add_file_handler(
    logger: Union[logging.Logger, str],
    log_file: Union[str, Path],
    level: Optional[Union[str, int]] = None,
    format_string: Optional[str] = None,
) -> None:
    """
    Add a file handler to an existing logger.

    Args:
        logger: Logger instance or logger name.
        log_file: Path to log file.
        level: Logging level for this handler. If None, uses logger's level.
        format_string: Format string for this handler. If None, uses DEFAULT_FORMAT.

    Example:
        >>> logger = setup_logging("myapp")
        >>> add_file_handler(logger, "debug.log", level="DEBUG")
    """
    if isinstance(logger, str):
        logger = logging.getLogger(logger)

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level or logger.level)

    fmt = format_string or DEFAULT_FORMAT
    file_handler.setFormatter(logging.Formatter(fmt))

    logger.addHandler(file_handler)


def disable_library_logging(library_name: str, level: int = logging.WARNING) -> None:
    """
    Reduce logging noise from third-party libraries.

    Args:
        library_name: Name of the library logger to quiet.
        level: Level to set (default: WARNING, to suppress INFO and DEBUG).

    Example:
        >>> # Quiet noisy libraries
        >>> disable_library_logging("urllib3")
        >>> disable_library_logging("requests")
    """
    logging.getLogger(library_name).setLevel(level)


# Module-level logger for internal use
_module_logger: Optional[logging.Logger] = None


def get_module_logger() -> logging.Logger:
    """
    Get the default logger for the eoflow package.

    Returns:
        Logger instance for eoflow.

    Example:
        >>> from eoflow.logging import get_module_logger
        >>> logger = get_module_logger()
        >>> logger.info("Using eoflow logger")
    """
    global _module_logger
    if _module_logger is None:
        _module_logger = setup_logging("eoflow")
    return _module_logger


# Example usage
if __name__ == "__main__":
    # Example 1: Basic console logging
    logger = setup_logging("example", level="DEBUG")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
