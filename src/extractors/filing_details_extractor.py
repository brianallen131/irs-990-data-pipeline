#!/usr/bin/env python3
"""
Extractor for independent contractor compensation details from IRS 990 filings.
"""

import argparse
from pathlib import Path
from .base_extractor import BaseIRSExtractor


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Extract filing details from IRS 990 filings'
    )
    parser.add_argument(
        '--force-process',
        action='store_true',
        help='Reprocess existing files (default: skip existing files)'
    )
    parser.add_argument(
        '--store-all-filing-data',
        action='store_true',
        help='Store all null filing data (default: skip storing data if all null)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1)'
    )

    args = parser.parse_args()

    # Configuration to edit
    raw_data_dir = Path("./data/raw_irs_data")
    output_dir = Path("./data/extracted_irs_data/filing_details")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = 'filing_details'

    FILING_METADATA_FIELDS = {
        'ein': 'EIN',
        'tax_yr': 'TaxYr',
        'form': 'ReturnTypeCd'
    }

    DETAILS_SUBROOTS = {
        '990': 'Filer',
        '990PF': 'Filer',
        '990EZ': 'Filer'
    }

    DETAILS_FIELDS = {
        'BusinessNameLine1Txt': {
            '990': 'BusinessNameLine1Txt',
            '990PF': 'BusinessNameLine1Txt',
            '990EZ': 'BusinessNameLine1Txt'
        },
        'AddressLine1Txt': {
            '990': 'AddressLine1Txt',
            '990PF': 'AddressLine1Txt',
            '990EZ': 'AddressLine1Txt'
        },
        'CityNm': {
            '990': 'CityNm',
            '990PF': 'CityNm',
            '990EZ': 'CityNm'
        },
        'ZIPCd': {
            '990': 'ZIPCd',
            '990PF': 'ZIPCd',
            '990EZ': 'ZIPCd'
        },
    }

    # Create and run extractor
    extractor = BaseIRSExtractor(
        raw_data_dir=raw_data_dir,
        output_dir=output_dir,
        output_prefix=output_prefix,
        force_process=args.force_process,
        store_all_filing_data=args.store_all_filing_data,
        FILING_METADATA_FIELDS=FILING_METADATA_FIELDS,
        DETAILS_SUBROOTS=DETAILS_SUBROOTS,
        DETAILS_FIELDS=DETAILS_FIELDS
    )
    extractor.process_all_zips(max_workers=args.max_workers)


if __name__ == "__main__":
    main()