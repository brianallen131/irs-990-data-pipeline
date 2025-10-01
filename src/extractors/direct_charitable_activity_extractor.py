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
        description='Extract direct charitable activity from IRS 990 filings'
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
    output_dir = Path("./data/extracted_irs_data/direct_charitable_activity_details")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = 'direct_charitable_activity_details'


    FILING_METADATA_FIELDS = {
        'ein': 'EIN',
        'tax_yr': 'TaxYr',
        'form': 'ReturnTypeCd'
    }

    DETAILS_SUBROOTS = {
        '990': '',
        '990PF': 'SummaryOfDirectChrtblActyGrp',
        '990EZ': ''
    }

    DETAILS_FIELDS = {
        'Description1Txt': {
            '990': '',
            '990PF': 'Description1Txt',
            '990EZ': ''
        },
        'Expenses1Amt': {
            '990': '',
            '990PF': 'Expenses1Amt',
            '990EZ': ''
        },


        'Description2Txt': {
            '990': '',
            '990PF': 'Description2Txt',
            '990EZ': ''
        },
        'Expenses2Amt': {
            '990': '',
            '990PF': 'Expenses2Amt',
            '990EZ': ''
        },

        'Description3Txt': {
            '990': '',
            '990PF': 'Description3Txt',
            '990EZ': ''
        },
        'Expenses3Amt': {
            '990': '',
            '990PF': 'Expenses3Amt',
            '990EZ': ''
        },

        'Description4Txt': {
            '990': '',
            '990PF': 'Description4Txt',
            '990EZ': ''
        },
        'Expenses4Amt': {
            '990': '',
            '990PF': 'Expenses4Amt',
            '990EZ': ''
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






