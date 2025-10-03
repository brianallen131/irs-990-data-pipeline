import pandas as pd

df_pri = pd.read_parquet('data/compiled_irs_data/program_related_investments_compiled.parquet')
df_od = pd.read_parquet('data/compiled_irs_data/organization_details_compiled.parquet')

df_pri[df_pri['ein']==453203840 ]

df_pri