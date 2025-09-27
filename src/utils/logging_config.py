#!/usr/bin/env python3
"""
Shared logging configuration for the IRS data pipeline.
"""

import logging
import sys

def setup_logger(name: str, level: int = logging.INFO,):
    """
        Set up a logger with consistent formatting across the project.

        Args:
            name: Logger name (typically __name__)
            level: Logging level (default: INFO)

        Returns:
            Configured logger instance
        """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def setup_project_logging(level: int = logging.INFO) -> None:
    """
    Set up logging for the entire project.

    Args:
        level: Base logging level for all modules
    """
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Set up console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)


def get_logger(name: str):
    """
    Get a logger with standard configuration.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return setup_logger(name)