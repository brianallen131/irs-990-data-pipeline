#!/usr/bin/env python3
"""
Base extractor class containing all common functionality for IRS data extraction.
"""

import pandas as pd
import zipfile_deflate64
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from ..utils.logging_config import get_logger

from threading import Lock
import concurrent.futures


class BaseIRSExtractor:
    """Base class for extracting data from IRS XML files."""

    def __init__(self,
                 raw_data_dir: Path,
                 output_dir: Path,
                 output_prefix: str,
                 FILING_METADATA_FIELDS: dict,
                 DETAILS_SUBROOTS: dict,
                 DETAILS_FIELDS: dict,
                 force_process: bool = False,
                 store_all_filing_data: bool = False
                 ):
        """
        Initialize the base extractor.

        Args:
            raw_data_dir: Directory containing zip files
            output_dir: Directory to save extracted CSV files
            output_prefix: Prefix of CSV file names within output_dir
            FILING_METADATA_FIELDS: Dictionary of filing metadata fields
            DETAILS_SUBROOTS: Mapping of form to subroot where extraction data is stored
            DETAILS_FIELDS: Mapping of form to subroot where extraction data is stored
            force_process: If True, reprocess existing files; if False, skip them
            store_all_filing_data: If True, store all data even if no information is stored
        """
        self.raw_data_dir = raw_data_dir
        self.output_dir = output_dir
        self.output_prefix = output_prefix
        self.force_process = force_process
        self.store_all_filing_data = store_all_filing_data
        self.FILING_METADATA_FIELDS = FILING_METADATA_FIELDS
        self.DETAILS_SUBROOTS = DETAILS_SUBROOTS
        self.DETAILS_FIELDS = DETAILS_FIELDS
        self.ns = '{http://www.irs.gov/efile}'

        # Use shared logging setup
        self.logger = get_logger(__name__)

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)


    def process_zip(self, zip_file: Path):
        # Read zipfile into memory
        with zipfile.ZipFile(zip_file, 'r') as zf:

            # Name of the file to be created to check if it exists, store for later for saving purposes
            output_file = self.output_dir / f'{self.output_prefix}__{zip_file.stem}.csv'

            # Skip file if already exists and force_process is turned off
            if output_file.exists() and not self.force_process:
                self.logger.info(f"Skipping {zip_file.name} - output already exists")
                return

            # Get list of XML files within each zip file
            xml_files = [name for name in zf.namelist() if name.endswith('.xml')]

            # Log metadata
            self.logger.info(f"Processing {len(xml_files)} XML files from {zip_file.name}")

            # List that will store rows
            zip_file_data = []

            # Loop over the XML files within each zip file
            for xml_file in xml_files:

                # Read the xml file and set root for parsing
                xml_content = zf.read(xml_file).decode('utf-8', errors='ignore')
                root = ET.fromstring(xml_content)

                # Store initial filing metadata
                filing_metadata = {
                    'filing_number': Path(xml_file).stem.split('_')[0],
                    'zip_name': zip_file.name
                }

                # Extract file metadata
                for col_name, xml_name in self.FILING_METADATA_FIELDS.items():
                    filing_metadata[col_name] = root.find(f'.//{self.ns}{xml_name}').text

                # Store form type for subroot and details parsing data
                form = filing_metadata['form']
                try:
                    subroots = root.findall(f'.//{self.ns}{self.DETAILS_SUBROOTS[form]}')
                except:
                    subroots = []

                # Loop over subroots in case there are lists of data to pull
                for subroot in subroots:
                    extraction_data = filing_metadata.copy()

                    # Boolean to see if any data was pulled
                    data_exists = False

                    # Extract data
                    for col_name, xml_names in self.DETAILS_FIELDS.items():
                        try:
                            xml_name = xml_names[form]
                            extraction_data[col_name] = subroot.find(f'.//{self.ns}{xml_name}').text
                            data_exists = True
                        except:
                            extraction_data[col_name] = None

                    # Only append data if exists or if explicitly told to store all filing data
                    if data_exists or self.store_all_filing_data:
                        zip_file_data.append(extraction_data)

            # Convert zip_file_data into a dataframe and store the csv to the output_file path
            df_zip_file_data = pd.DataFrame(zip_file_data)
            df_zip_file_data.to_csv(output_file, index=False)

    def process_all_zips(self, max_workers: int = 1):
        # Get list of zip files from the raw data folder
        zip_files = list(self.raw_data_dir.glob('*.zip'))

        if max_workers == 1:
            self.logger.info(f"Processing {len(zip_files)} zip files.")

            # Loop through zips and process sequentially
            for zip_file in zip_files:
                self.process_zip(zip_file)
        else:
            self.logger.info(f"Processing {len(zip_files)} zip files with {max_workers} workers.")

            # Create a thread pool to execute extraction in parallel across worker threads
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(executor.map(self.process_zip, zip_files))
