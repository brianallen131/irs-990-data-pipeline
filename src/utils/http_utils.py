#!/usr/bin/env python3
"""
HTTP utilities for the IRS data pipeline.
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional


def create_session(
        user_agent: str = 'Mozilla/5.0 (compatible; IRS-990-Downloader/1.0)',
        max_retries: int = 3,
        backoff_factor: float = 1.0
):
    """
    Create a robust HTTP session with retry logic and proper headers.

    Args:
        user_agent: User-Agent string to use
        max_retries: Maximum number of retries for failed requests
        backoff_factor: Backoff factor for retry delays

    Returns:
        Configured requests.Session object
    """
    session = requests.Session()

    # Set up retry strategy
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )

    # Create adapter with retry strategy
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Set headers
    session.headers.update({
        'User-Agent': user_agent,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    })

    return session


def download_with_progress(
        session: requests.Session,
        url: str,
        filepath: str,
        chunk_size: int = 8192,
        timeout: int = 60
):
    """
    Download a file with progress tracking.

    Args:
        session: Requests session to use
        url: URL to download
        filepath: Local path to save file
        chunk_size: Size of chunks to download
        timeout: Request timeout

    Returns:
        True if successful, False otherwise
    """
    try:
        response = session.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

        return True

    except (requests.RequestException, IOError):
        return False