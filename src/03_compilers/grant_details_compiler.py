"""
Compiler for grant details data.
Filters against the distinct filings from organization_details_compiled.parquet
"""
import pandas as pd
from pathlib import Path


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


def load_distinct_filings(compiled_data_dir: Path) -> pd.DataFrame:
    """Load distinct filings from the main organization details file."""
    df_org = pd.read_parquet(
        compiled_data_dir / 'organization_details_compiled.parquet'
    )

    distinct_filing_cols = ['filing_number', 'zip_name', 'ein', 'tax_yr', 'form']
    return df_org[distinct_filing_cols].drop_duplicates()


def main():
    # Define data types
    gd_data_dtypes = {
        'filing_number': int,
        'zip_name': str,
        'ein': int,
        'tax_yr': int,
        'form': str,
        'BusinessNameLine1Txt': str,
        'BusinessNameLine2Txt': str,
        'RecipientPersonNm': str,
        'RecipientEIN': 'Int64',
        'AddressLine1Txt': str,
        'CityNm': str,
        'StateAbbreviationCd': str,
        'ZIPCd': str,
        'GrantOrContributionPurposeTxt': str,
        'CashGrantAmt': float
    }

    distinct_filing_cols = ['filing_number', 'zip_name', 'ein', 'tax_yr', 'form']

    # Load grant details
    print("Loading grant details...")
    gd_data_dir = Path("data/extracted_irs_data/grant_details")
    df_gd = load_csv_files(gd_data_dir, gd_data_dtypes)

    # Combine business name and person name into single recipient field
    df_gd['RecipientNm'] = df_gd['BusinessNameLine1Txt'].combine_first(
        df_gd['RecipientPersonNm']
    )

    # Select and rename columns
    df_gd_clean = df_gd[[
        'filing_number', 'zip_name', 'ein', 'tax_yr', 'form',
        'RecipientEIN', 'RecipientNm', 'AddressLine1Txt',
        'CityNm', 'StateAbbreviationCd', 'ZIPCd',
        'GrantOrContributionPurposeTxt', 'CashGrantAmt'
    ]].copy()

    df_gd_clean.columns = [
        'filing_number', 'zip_name', 'ein', 'tax_yr', 'form',
        'recipient_ein', 'recipient_name', 'recipient_address',
        'recipient_city', 'recipient_state', 'recipient_zip',
        'grant_purpose', 'grant_amt'
    ]

    # Standardize ZIP codes to 5 digits
    df_gd_clean['recipient_zip'] = (
        df_gd_clean['recipient_zip'].astype(str).str[:5].str.zfill(5)
    )

    # Filter to only distinct filings from main organization file
    print("Filtering to distinct filings...")
    compiled_data_dir = Path("data/compiled_irs_data")
    distinct_filings = load_distinct_filings(compiled_data_dir)

    df_gd_final = df_gd_clean.merge(
        distinct_filings, how='inner', on=distinct_filing_cols
    )

    # Save output
    print("Saving compiled grant details...")
    output_dir = Path("data/compiled_irs_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    df_gd_final.to_parquet(
        output_dir / 'grant_details_compiled.parquet',
        engine='fastparquet'
    )

    print(f"Compiled {len(df_gd_final):,} grant records")

    return df_gd_final


if __name__ == "__main__":
    df_grants = main()