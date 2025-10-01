import pandas as pd
from pathlib import Path

cga_data_dtypes = {
    'filing_number': int,
    'zip_name': str,
    'ein': int,
    'tax_yr': int,
    'form':str,
    'contribution_amt': float,
    'grant_amt': float,
    'managed_assets_at_eoy':float
}

cga_data_dir = Path("data/extracted_irs_data/contributions_grants_assets")
cga_files = list(cga_data_dir.glob('*.csv'))
dataframes = []
for grant_file in grant_files:
    dataframes.append(pd.read_csv(grant_file,dtype=cga_data_dtypes))

df_cga = pd.concat(dataframes).fillna(0)

df_deduped = df_cga.sort_values(by='filing_number', ascending=False)

# Drop duplicates based on the grouping columns, keeping the first row (which is now the highest filing_number)
df_deduped.drop_duplicates(subset=['ein', 'tax_yr', 'form'], keep='first',inplace=True)

output_dir = Path("./data/cleaned_irs_data/")
output_dir.mkdir(parents=True, exist_ok=True)

df_deduped.sort_values(by=['managed_assets_at_eoy'], ascending=False, inplace=True)


output_path = output_dir / 'clean_contributions_grants_assets.csv'
df_deduped.to_csv(output_path)