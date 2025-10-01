import pandas as pd
from pathlib import Path

daf_data_dtypes = {
    'filing_number': int,
    'zip_name': str,
    'ein': int,
    'tax_yr': int,
    'form':str,
    'daf_contribution_amt': float,
    'daf_grant_amt': float,
    'daf_managed_assets_at_eoy':float,
    'daf_count_at_eoy':float
}

daf_data_dir = Path("data/extracted_irs_data/donor_advised_fund_details")
daf_files = list(daf_data_dir.glob('*.csv'))
dataframes = []
for file in daf_files:
    try:
        df = pd.read_csv(file,dtype=daf_data_dtypes)
        if not df.empty:
            dataframes.append(df)
    except:
        print(f"Couldn't read {file}")
    # dataframes.append(pd.read_csv(file,dtype=daf_data_dtypes))

df_daf = pd.concat(dataframes).fillna(0)

df_deduped = df_daf.sort_values(by='filing_number', ascending=False)

# Drop duplicates based on the grouping columns, keeping the first row (which is now the highest filing_number)
df_deduped.drop_duplicates(subset=['ein', 'tax_yr', 'form'], keep='first',inplace=True)

output_dir = Path("./data/cleaned_irs_data/")
output_dir.mkdir(parents=True, exist_ok=True)

df_deduped.sort_values(by=['daf_managed_assets_at_eoy'], ascending=False, inplace=True)


output_path = output_dir / 'clean_donor_advised_fund_details.csv'
df_deduped.to_csv(output_path)
