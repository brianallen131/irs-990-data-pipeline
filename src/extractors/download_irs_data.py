#!/usr/bin/env python3
"""
IRS Form 990 Data Downloader

Downloads zip files from IRS 990 datastore.
Process:
    1. Scrapes all .zip and .csv file names from IRS website
    2. Downloads files with retry logic and progress tracking
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Import shared utilities
from src.config.settings import DownloadConfig
from src.utils.logging_config import get_logger
from src.utils.http_utils import create_session


class IRSDataDownloader:
    """Downloads and manages IRS Form 990 data files."""

    def __init__(self, config: DownloadConfig = None):
        self.config = config or DownloadConfig()
        self.session = create_session(
            user_agent=self.config.user_agent,
            max_retries=self.config.max_retries
        )
        self.logger = get_logger(__name__)

        # Create data directory
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

    def get_download_urls(self) -> Tuple[List[str], List[str]]:
        """
        Extract ZIP and CSV download URLs from IRS website.

        Returns:
            Tuple of (zip_urls, csv_urls)
        """
        try:
            response = self.session.get(
                self.config.base_url,
                timeout=self.config.request_timeout
            )
            response.raise_for_status()
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch URL list: {e}")
            raise

        soup = BeautifulSoup(response.content, 'html.parser')
        zip_urls = []
        csv_urls = []

        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(self.config.base_url, href)

            if href.endswith('.zip'):
                zip_urls.append(full_url)
            elif href.endswith('.csv'):
                csv_urls.append(full_url)

        self.logger.info(f"Found {len(zip_urls)} ZIP files and {len(csv_urls)} CSV files")
        return zip_urls, csv_urls

    def _file_exists_and_valid(self, filepath: Path) -> bool:
        """Check if file exists and is valid."""
        return filepath.exists() and filepath.stat().st_size > 0

    def _download_file(self, url: str, force_redownload: bool = False) -> bool:
        """
        Download a single file with progress tracking and error handling.

        Args:
            url: URL to download
            force_redownload: Whether to redownload existing files

        Returns:
            True if successful, False otherwise
        """
        filename = Path(urlparse(url).path).name
        filepath = self.config.data_dir / filename

        # Check if file already exists
        if not force_redownload and self._file_exists_and_valid(filepath):
            self.logger.info(f"Skipping {filename} (already exists)")
            return True

        # For CSV files, remove existing file to ensure fresh download
        if filepath.suffix.lower() == '.csv' and filepath.exists():
            filepath.unlink()

        for attempt in range(self.config.max_retries):
            try:
                response = self.session.get(
                    url,
                    stream=True,
                    timeout=self.config.request_timeout
                )
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))

                with open(filepath, 'wb') as f, tqdm(
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        desc=filename,
                        disable=total_size == 0
                ) as pbar:

                    for chunk in response.iter_content(chunk_size=self.config.chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

                self.logger.info(f"Successfully downloaded {filename}")
                return True

            except (requests.RequestException, IOError) as e:
                self.logger.warning(
                    f"Download attempt {attempt + 1} failed for {filename}: {e}"
                )
                if filepath.exists():
                    filepath.unlink()  # Remove partial file

                if attempt == self.config.max_retries - 1:
                    self.logger.error(f"Failed to download {filename} after {self.config.max_retries} attempts")
                    return False

        return False

    def download_all_files(self,
                           force_redownload: bool = False,
                           parallel: bool = True) -> Tuple[int, int]:
        """
        Download all ZIP and CSV files.

        Args:
            force_redownload: Whether to redownload existing files
            parallel: Whether to use parallel downloads

        Returns:
            Tuple of (successful_downloads, total_files)
        """
        zip_urls, csv_urls = self.get_download_urls()
        all_urls = zip_urls + csv_urls

        if not all_urls:
            self.logger.warning("No files found to download")
            return 0, 0

        successful_downloads = 0

        if parallel and len(all_urls) > 1:
            # Parallel downloads
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_url = {
                    executor.submit(self._download_file, url, force_redownload): url
                    for url in all_urls
                }

                for future in as_completed(future_to_url):
                    if future.result():
                        successful_downloads += 1
        else:
            # Sequential downloads
            for url in all_urls:
                if self._download_file(url, force_redownload):
                    successful_downloads += 1

        self.logger.info(
            f"Download complete: {successful_downloads}/{len(all_urls)} files successful"
        )
        return successful_downloads, len(all_urls)

    def get_file_inventory(self) -> dict:
        """Get inventory of downloaded files."""
        if not self.config.data_dir.exists():
            return {}

        inventory = {}
        for file_path in self.config.data_dir.iterdir():
            if file_path.is_file():
                stat = file_path.stat()
                inventory[file_path.name] = {
                    'size_bytes': stat.st_size,
                    'size_mb': round(stat.st_size / (1024 * 1024), 2),
                    'modified': stat.st_mtime
                }

        return inventory


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(description="Download IRS Form 990 data files")
    parser.add_argument(
        '--force',
        action='store_true',
        help="Force redownload of existing files"
    )
    parser.add_argument(
        '--sequential',
        action='store_true',
        help="Use sequential instead of parallel downloads"
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data/raw_irs_data'),
        help="Directory to store downloaded files"
    )
    parser.add_argument(
        '--inventory',
        action='store_true',
        help="Show inventory of existing files and exit"
    )

    args = parser.parse_args()

    config = DownloadConfig(data_dir=args.data_dir)
    downloader = IRSDataDownloader(config)

    if args.inventory:
        inventory = downloader.get_file_inventory()
        if inventory:
            print("\nFile Inventory:")
            print("-" * 50)
            for filename, info in inventory.items():
                print(f"{filename}: {info['size_mb']} MB")
        else:
            print("No files found in data directory")
        return

    try:
        successful, total = downloader.download_all_files(
            force_redownload=args.force,
            parallel=not args.sequential
        )

        if successful == total:
            print(f"\n✅ Successfully downloaded all {total} files")
        else:
            print(f"\n⚠️ Downloaded {successful}/{total} files")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()