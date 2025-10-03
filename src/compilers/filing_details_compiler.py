"""
Main compiler for IRS filing details.
Creates the master list of distinct filings used by all other compilers.
"""
import pandas as pd
from pathlib import Path

# Configure pandas display
pd.options.display.float_format = '{:,.2f}'.format
pd.reset_option('display.float_format')


def load_csv_files(data_dir: Path, dtypes: dict) -> pd.DataFrame:
    """Load and concatenate all CSV files from a directory."""
    files = list(data_dir.glob('*.csv'))
    dataframes = []

    for file in files:
        try:
            df = pd.read_csv(file, dtype=dtypes)
            if not df.empty:
                dataframes.append(df)
        except Exception as e:
            print(f"Couldn't read {file}: {e}")

    return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()


def deduplicate_filings(df: pd.DataFrame, sort_col: str, subset_cols: list) -> pd.DataFrame:
    """
    Deduplicate filings, keeping the most recent filing.
    Sorts by sort_col descending and keeps first occurrence.
    """
    return df.sort_values(by=sort_col, ascending=False).drop_duplicates(
        subset=subset_cols, keep='first'
    )


def main():
    # Define merge columns
    merging_cols = ['filing_number', 'zip_name', 'ein', 'tax_yr', 'form']

    # ==================== Filing Details ====================
    print("Loading filing details...")
    fd_data_dir = Path("data/extracted_irs_data/filing_details")
    fd_data_dtypes = {
        'filing_number': int,
        'zip_name': str,
        'ein': int,
        'tax_yr': int,
        'form': str,
        'BusinessNameLine1Txt': str,
        'AddressLine1Txt': str,
        'CityNm': str,
        'ZIPCd': str
    }
    df_fd = load_csv_files(fd_data_dir, fd_data_dtypes)

    # Deduplicate: keep most recent filing for each (ein, tax_yr, form)
    df_fd_dedup = deduplicate_filings(df_fd, 'filing_number', ['ein', 'tax_yr', 'form'])

    # Rename columns for clarity
    df_fd_dedup.columns = [
        'filing_number', 'zip_name', 'ein', 'tax_yr', 'form',
        'foundation_name', 'foundation_address', 'foundation_city', 'foundation_zip'
    ]

    # Standardize ZIP codes to 5 digits
    df_fd_dedup['foundation_zip'] = (
        df_fd_dedup['foundation_zip'].astype(str).str[:5].str.zfill(5)
    )

    # ==================== Contributions, Grants, Assets ====================
    print("Loading contributions, grants, and assets...")
    cga_data_dir = Path("data/extracted_irs_data/contributions_grants_assets")
    cga_data_dtypes = {
        'filing_number': int,
        'zip_name': str,
        'ein': int,
        'tax_yr': int,
        'form': str,
        'contribution_amt': float,
        'grant_amt': float,
        'managed_assets_at_eoy': float
    }
    df_cga = load_csv_files(cga_data_dir, cga_data_dtypes).fillna(0)

    df_cga_dedup = deduplicate_filings(
        df_cga, 'managed_assets_at_eoy',
        ['filing_number', 'ein', 'tax_yr', 'form']
    )

    # ==================== Donor Advised Fund Details ====================
    print("Loading donor advised fund details...")
    daf_data_dir = Path("data/extracted_irs_data/donor_advised_fund_details")
    daf_data_dtypes = {
        'filing_number': int,
        'zip_name': str,
        'ein': int,
        'tax_yr': int,
        'form': str,
        'daf_contribution_amt': float,
        'daf_grant_amt': float,
        'daf_managed_assets_at_eoy': float,
        'daf_count_at_eoy': float
    }
    df_daf = load_csv_files(daf_data_dir, daf_data_dtypes).fillna(0)

    df_daf_dedup = deduplicate_filings(
        df_daf, 'daf_managed_assets_at_eoy',
        ['filing_number', 'ein', 'tax_yr', 'form']
    )

    # ==================== Exemption and Service Description ====================
    print("Loading exemption and service descriptions...")
    eas_data_dir = Path("data/extracted_irs_data/exemption_and_service_description")
    eas_data_dtypes = {
        'filing_number': int,
        'zip_name': str,
        'ein': int,
        'tax_yr': int,
        'form': str,
        'PrimaryExemptPurposeTxt': str,
        'DescriptionProgramSrvcAccomTxt': str
    }
    df_eas = load_csv_files(eas_data_dir, eas_data_dtypes)


    df_eas_dedup = df_eas.sort_values(
        'PrimaryExemptPurposeTxt', ascending=False
    ).drop_duplicates(subset=['filing_number', 'ein', 'tax_yr', 'form'], keep='first')

    df_eas_dedup.columns = [
        'filing_number', 'zip_name', 'ein', 'tax_yr', 'form',
        'organization_mission_pt1_990_only', 'organization_mission_pt2_990_only'
    ]

    # ==================== Merge All Data ====================
    print("Merging all datasets...")
    df_organization_details = (
        df_fd_dedup
        .merge(df_eas_dedup, how='left', on=merging_cols)
        .merge(df_cga_dedup, how='left', on=merging_cols)
        .merge(df_daf_dedup, how='left', on=merging_cols)
    )

    # Fill missing financial values with 0
    cols_to_fill = [
        'contribution_amt', 'grant_amt', 'managed_assets_at_eoy',
        'daf_contribution_amt', 'daf_grant_amt',
        'daf_managed_assets_at_eoy', 'daf_count_at_eoy'
    ]
    df_organization_details[cols_to_fill] = df_organization_details[cols_to_fill].fillna(0)

    # ==================== Save Output ====================
    print("Saving compiled data...")
    output_dir = Path("data/compiled_irs_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    df_organization_details.to_parquet(
        output_dir / 'organization_details_compiled.parquet',
        engine='fastparquet'
    )

    print(f"Compiled {len(df_organization_details):,} organization records")

    return df_organization_details


if __name__ == "__main__":
    df_organization_details = main()
