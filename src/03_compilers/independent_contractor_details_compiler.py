"""
Compiler for independent contractor details.
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
    icd_data_dtypes = {
        'filing_number': int,
        'zip_name': str,
        'ein': int,
        'tax_yr': int,
        'form': str,
        'PersonNm': str,
        'BusinessNameLine1Txt': str,
        'AddressLine1Txt': str,
        'CityNm': str,
        'StateAbbreviationCd': str,
        'ZIPCd': str,
        'ServicesDesc': str,
        'CompensationAmt': float
    }

    distinct_filing_cols = ['filing_number', 'zip_name', 'ein', 'tax_yr', 'form']

    # Load contractor details
    print("Loading independent contractor details...")
    icd_data_dir = Path("data/extracted_irs_data/independent_contractor_details")
    df_icd = load_csv_files(icd_data_dir, icd_data_dtypes)

    # Combine business name and person name into single entity field
    df_icd['EntityNm'] = df_icd['BusinessNameLine1Txt'].combine_first(
        df_icd['PersonNm']
    )

    # Select and rename columns
    df_icd_clean = df_icd[[
        'filing_number', 'zip_name', 'ein', 'tax_yr', 'form',
        'EntityNm', 'AddressLine1Txt', 'CityNm',
        'StateAbbreviationCd', 'ZIPCd', 'ServicesDesc', 'CompensationAmt'
    ]].copy()

    df_icd_clean.columns = [
        'filing_number', 'zip_name', 'ein', 'tax_yr', 'form',
        'contractor_name', 'contractor_address', 'contractor_city',
        'contractor_state', 'contractor_zip', 'service_description',
        'compensation_amt'
    ]

    # Standardize ZIP codes to 5 digits
    df_icd_clean['contractor_zip'] = (
        df_icd_clean['contractor_zip'].astype(str).str[:5].str.zfill(5)
    )

    # Filter to only distinct filings from main organization file
    print("Filtering to distinct filings...")
    compiled_data_dir = Path("data/compiled_irs_data")
    distinct_filings = load_distinct_filings(compiled_data_dir)

    df_icd_final = df_icd_clean.merge(
        distinct_filings, how='inner', on=distinct_filing_cols
    )

    # Save output
    print("Saving compiled contractor details...")
    output_dir = Path("data/compiled_irs_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    df_icd_final.to_parquet(
        output_dir / 'independent_contractor_details_compiled.parquet',
        engine='fastparquet'
    )

    print(f"Compiled {len(df_icd_final):,} contractor records")

    return df_icd_final


if __name__ == "__main__":
    df_contractors = main()