"""
logger.py
====================
This module provides a singleton Logger class that configures
a project-wide logging system. It ensures consistent, centralized
logging across all modules and prevents multiple logger instances.

Features:
- Singleton pattern (only one instance per process)
- Logs to both console and rotating file
- Auto-creates the logs directory
- Accepts DEBUG_MODE flag from caller
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


class Logger:
    """Singleton logger for the entire project.

    This class ensures that only one instance of the logger exists,
    and it can be safely imported anywhere in the project
    without creating duplicate log handlers.

    Example:
        >>> from src.utils.logger import Logger
        >>> logger = Logger(debug=True).get_logger(__name__)
        >>> logger.info("Spotify dataset successfully loaded.")
    """

    _instance = None  # Holds the singleton instance

    def __new__(cls, debug: bool = False):
        """Create a single logger instance if it doesn't already exist."""
        if cls._instance is None:
            print("Initialized logger...")
            cls._instance = super().__new__(cls)

            # Get project root (two levels up from current file)
            project_root = Path(__file__).resolve().parents[2]

            # Create logs folder if it doesn't exist
            log_dir = project_root / "logs"
            log_dir.mkdir(exist_ok=True)

            # Ensure log file exists
            log_file = log_dir / "app.log"
            log_file.touch(exist_ok=True)

            # Log format and date format
            log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
            date_format = "%Y-%m-%d %H:%M:%S"

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

            # File handler (rotating to avoid infinite growth, e.g., 5MB × 3 backups)
            file_handler = RotatingFileHandler(
                log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
            )
            file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

            # Root logger setup
            root_logger = logging.getLogger()

            # Avoid adding duplicate handlers if this logger is re-imported
            if not root_logger.handlers:
                root_logger.setLevel(logging.DEBUG if debug else logging.INFO)
                root_logger.addHandler(console_handler)
                root_logger.addHandler(file_handler)

        return cls._instance
    

    def get_logger(self, name: str) -> logging.Logger:
        """Return a logger instance with the given name.

        Args:
            name (str): The module name, typically __name__.

        Returns:
            logging.Logger: Configured logger instance.
        """
        return logging.getLogger(name)
