import pandas as pd
import matplotlib as mpl

df_icd = pd.read_parquet('data/compiled_irs_data/independent_contractor_details_compiled.parquet')
df_od = pd.read_parquet('data/compiled_irs_data/organization_details_compiled.parquet')

tmp = df_icd[df_icd['contractor_name'].str.contains('aperio',case=False,na=False)]
tmp = df_icd[df_icd['contractor_name'].str.contains('cambridge ass',case=False,na=False)]
tmp.sort_values('compensation_amt',ascending=False,inplace=True)

tmp2 = tmp.groupby(['tax_yr'])['compensation_amt'].sum()

df_icd_investments = df_icd[
    (df_icd['service_description'].str.contains('investment',case=False,na=False)) &
    (df_icd['tax_yr']==2023)
]
top_investment_contractors = df_icd_investments.groupby(['contractor_name'])['compensation_amt'].sum().sort_values(ascending=False)


total_investment = df_icd_investments.groupby(['tax_yr'])['compensation_amt'].sum()


def format_large_numbers(x):
    """Format numbers with M, B, T suffixes"""
    if pd.isna(x):
        return x

    abs_x = abs(x)

    if abs_x >= 1e12:
        return f"${x / 1e12:.2f}T"
    elif abs_x >= 1e9:
        return f"${x / 1e9:.2f}B"
    elif abs_x >= 1e6:
        return f"${x / 1e6:.2f}M"
    else:
        return f"${x:.2f}"

total_investment.apply(format_large_numbers)

df_icd[df_icd['ein']==453203840 ]