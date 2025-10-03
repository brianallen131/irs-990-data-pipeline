"""
Compiler for program-related investments.
Transforms wide format into long format with one row per investment.
Creates a third category for "All other program-related investments" when present.
Filters against the distinct filings from organization_details_compiled.parquet
"""
import pandas as pd
from pathlib import Path
import numpy as np


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


def reshape_to_long_format(df: pd.DataFrame, id_vars: list) -> pd.DataFrame:
    """
    Reshape from wide format to long format.
    Creates Description3/Expenses3 from AllOtherProgramRltdInvstTotAmt when present.
    Excludes generic placeholder descriptions.
    """
    # Create third description/expense pair from "all other" field
    df['Expenses3Amt'] = df['AllOtherProgramRltdInvstTotAmt']
    df['Description3Txt'] = np.where(
        df['AllOtherProgramRltdInvstTotAmt'].fillna(0) != 0,
        "All other program-related investments",
        None
    )

    # Melt descriptions
    df_desc = pd.melt(
        df,
        id_vars=id_vars,
        value_vars=['Description1Txt', 'Description2Txt', 'Description3Txt'],
        var_name='desc_num',
        value_name='DescriptionTxt'
    )

    # Melt expenses
    df_exp = pd.melt(
        df,
        id_vars=id_vars,
        value_vars=['Expenses1Amt', 'Expenses2Amt', 'Expenses3Amt'],
        var_name='exp_num',
        value_name='ExpensesAmt'
    )

    # Extract numeric index from variable names
    df_desc['num'] = df_desc['desc_num'].str.extract(r'(\d+)')
    df_exp['num'] = df_exp['exp_num'].str.extract(r'(\d+)')

    # Filter out generic placeholder descriptions
    df_desc['desc_clean'] = df_desc['DescriptionTxt'].str.upper().str.strip()

    exclude_patterns = [
        'SEE STATEMENT',
        'SEE GENERAL EXPLANATION',
        '(SEE STATEMENT)',
        'SEE ATTACHMENT',
        'NONE'
    ]

    mask = df_desc['desc_clean'].notna()
    for pattern in exclude_patterns:
        mask &= ~df_desc['desc_clean'].str.startswith(pattern).fillna(False)

    df_desc = df_desc[mask].drop(columns=['desc_clean'])

    # Merge descriptions and expenses
    df_long = pd.merge(
        df_desc, df_exp,
        on=id_vars + ['num'],
        how='inner'
    )

    # Clean up temporary columns
    df_long = df_long.drop(columns=['desc_num', 'exp_num', 'num'])
    df_long['ExpensesAmt'] = df_long['ExpensesAmt'].astype(float).fillna(0)

    return df_long


def main():
    # Define data types
    pri_data_dtypes = {
        'filing_number': int,
        'zip_name': str,
        'ein': int,
        'tax_yr': int,
        'form': str,
        'Description1Txt': str,
        'Expenses1Amt': float,
        'Description2Txt': str,
        'Expenses2Amt': float,
        'AllOtherProgramRltdInvstTotAmt': float,
        'TotalAmt': float
    }

    id_vars = ['filing_number', 'zip_name', 'ein', 'tax_yr', 'form']

    # Load program-related investment details
    print("Loading program-related investment details...")
    pri_data_dir = Path("data/extracted_irs_data/program_related_investments")
    df_pri = load_csv_files(pri_data_dir, pri_data_dtypes)

    # Reshape from wide to long format
    print("Reshaping to long format...")
    df_pri_long = reshape_to_long_format(df_pri, id_vars)

    # Filter to only distinct filings from main organization file
    print("Filtering to distinct filings...")
    compiled_data_dir = Path("data/compiled_irs_data")
    distinct_filings = load_distinct_filings(compiled_data_dir)

    df_pri_final = df_pri_long.merge(
        distinct_filings, how='inner', on=id_vars
    )

    # Save output
    print("Saving compiled program-related investment details...")
    output_dir = Path("data/compiled_irs_data")
    output_dir.mkdir(parents=True, exist_ok=True)

    df_pri_final.to_parquet(
        output_dir / 'program_related_investments_compiled.parquet',
        engine='fastparquet'
    )

    print(f"Compiled {len(df_pri_final):,} program-related investment records")

    return df_pri_final


if __name__ == "__main__":
    df_pri = main()