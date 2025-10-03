import pandas as pd

df_gd = pd.read_parquet('data/compiled_irs_data/grant_details_compiled.parquet')
df_od = pd.read_parquet('data/compiled_irs_data/organization_details_compiled.parquet')

df_od['has_daf_grants']=df_od['daf_grant_amt'] > 0

merge_cols = [
    'filing_number',
    'zip_name',
    'ein',
    'tax_yr',
    'form'
]

df.groupby(['recipient_ein'])['grant_amt'].sum().sort_values(ascending=False)


tmp = df[df['recipient_ein']==911673170]
tmp2 = tmp.merge(df_od[merge_cols+['has_daf_grants']], on=merge_cols, how='left')

df_od[df_od['ein']==453203840 ]
df_gd[df_gd['ein']==453203840 ]