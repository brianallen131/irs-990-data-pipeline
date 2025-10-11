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
        description='Extract contributions, grants and assets details from IRS 990 filings'
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
    output_dir = Path("./data/extracted_irs_data/contributions_grants_assets")   # Change: File Path for final data
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = 'contributions_grants_assets'                                # Change: csv prefix for each file

    FILING_METADATA_FIELDS = {
        'ein': 'EIN',
        'tax_yr': 'TaxYr',
        'form': 'ReturnTypeCd'
    }

    DETAILS_SUBROOTS = {       # Change: Subroot where elem names to pull is unique
        '990': 'IRS990',
        '990PF': 'IRS990PF',
        '990EZ': 'IRS990EZ'
    }

    DETAILS_FIELDS = {         # Change: Column name and elem names to pull
        # Total contributions
        'contribution_amt': {                              # Column to store data under
                '990': 'PYContributionsGrantsAmt',         # XML elem name to extract
                '990PF': 'ContriRcvdRevAndExpnssAmt',      # XML elem name to extract
                '990EZ': 'ContributionsGiftsGrantsEtcAmt'  # ...
        },
        # Total grants
        'grant_amt': {
            '990': 'PYGrantsAndSimilarPaidAmt',
            '990PF': 'ContriPaidRevAndExpnssAmt',
            '990EZ': 'GrantsAndSimilarAmountsPaidAmt'
        },
        # Total assets
        'managed_assets_at_eoy': {
            '990': 'NetAssetsOrFundBalancesEOYAmt',
            '990PF': 'TotNetAstOrFundBalancesEOYAmt',
            '990EZ': 'NetAssetsOrFundBalancesEOYAmt'
        }
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