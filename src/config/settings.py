#!/usr/bin/env python3
"""
Common configuration settings for the IRS data pipeline.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any


@dataclass
class DownloadConfig:
    """Configuration settings for the IRS data downloader."""
    base_url: str = 'https://www.irs.gov/charities-non-profits/form-990-series-downloads'
    data_dir: Path = Path('data/raw_irs_data')
    chunk_size: int = 8192
    request_timeout: int = 60
    max_workers: int = 4
    max_retries: int = 3
    user_agent: str = 'Mozilla/5.0 (compatible; IRS-990-Downloader/1.0)'


@dataclass
class ExtractionConfig:
    """Configuration settings for data extraction."""
    raw_data_dir: Path = Path('data/raw_irs_data')
    output_base_dir: Path = Path('data/extracted_irs_data')
    namespace: str = '{http://www.irs.gov/efile}'

    @property
    def contributions_output_dir(self) -> Path:
        """Directory for contributions, grants, and assets output."""
        return self.output_base_dir / 'contributions_grants_assets'

    @property
    def grants_output_dir(self) -> Path:
        """Directory for grant details output."""
        return self.output_base_dir / 'grant_details'


@dataclass
class ProjectConfig:
    """Main project configuration."""
    project_root: Path = Path('.')

    # Sub-configurations
    download: DownloadConfig = None
    extraction: ExtractionConfig = None

    def __post_init__(self):
        """Initialize sub-configurations if not provided."""
        if self.download is None:
            self.download = DownloadConfig()
        if self.extraction is None:
            self.extraction = ExtractionConfig()


# IRS Form 990 field mappings (shared across 02_extractors)
FILING_METADATA_FIELDS = {
    'ein': 'EIN',
    'tax_yr': 'TaxYr',
    'form': 'ReturnTypeCd'
}

CONTRIBUTIONS_GRANTS_ASSETS_FIELDS = {
    # Total contributions
    'contribution_amt': {
        '990': 'PYContributionsGrantsAmt',
        '990PF': 'ContriRcvdRevAndExpnssAmt',
        '990EZ': 'GrossReceiptsAmt'
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
        '990PF': 'CashEOYAmt',
        '990EZ': 'NetAssetsOrFundBalancesEOYAmt'
    },
    # DAF-specific fields (990 only)
    'daf_contribution_amt': {
        '990': 'DonorAdvisedFundsContriAmt'
    },
    'daf_grant_amt': {
        '990': 'DonorAdvisedFundsGrantsAmt'
    },
    'daf_managed_assets_at_eoy': {
        '990': 'DonorAdvisedFundsVlEOYAmt'
    },
    'daf_count_at_eoy': {
        '990': 'DonorAdvisedFundsHeldCnt'
    }
}

GRANT_DETAILS_FIELDS = {
    'BusinessNameLine1Txt': 'BusinessNameLine1Txt',
    'AddressLine1Txt': 'AddressLine1Txt',
    'CityNm': 'CityNm',
    'StateAbbreviationCd': 'StateAbbreviationCd',
    'ZIPCd': 'ZIPCd',
    'GrantOrContributionPurposeTxt': 'GrantOrContributionPurposeTxt',
    'Amt': 'Amt'
}