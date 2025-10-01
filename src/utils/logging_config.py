#!/usr/bin/env python3
"""
Shared logging configuration for the IRS data pipeline.
"""

import logging
import sys

def get_logger(name: str, level: int = logging.INFO,):
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