import pandas as pd
from pathlib import Path

icd_data_dtypes = {
    'filing_number': int,
    'zip_name': str,
    'ein': int,
    'tax_yr': int,
    'form':str,
    'PersonNm': str,
    'BusinessNameLine1Txt': str,
    'AddressLine1Txt': str,
    'CityNm': str,
    'StateAbbreviationCd': str,
    'ZIPCd': str,
    'ServicesDesc': str,
    'CompensationAmt': float
}

icd_data_dir = Path("data/extracted_irs_data/independent_contractor_details")
icd_files = list(icd_data_dir.glob('*.csv'))
dataframes = []
for file in icd_files:
    try:
        df = pd.read_csv(file,dtype=icd_data_dtypes)
        if not df.empty:
            dataframes.append(df)
    except:
        print(f"Couldn't read {file}")
    # dataframes.append(pd.read_csv(file,dtype=daf_data_dtypes))

df_icd = pd.concat(dataframes)

df_icd['EntityNm'] = df_icd['BusinessNameLine1Txt'].combine_first(df_icd['PersonNm'])

df_icd_dedup = df_icd[['filing_number', 'zip_name', 'ein', 'tax_yr', 'form', 'EntityNm', 'AddressLine1Txt', 'CityNm',
       'StateAbbreviationCd', 'ZIPCd', 'ServicesDesc', 'CompensationAmt']]

tmp =  df_icd_dedup[df_icd_dedup['EntityNm']=='APERIO']
tmp =  df_icd_dedup[df_icd_dedup['ServicesDesc']=='INVESTMENT ADVICE'].groupby(['EntityNm'])['CompensationAmt'].sum().sort_values(ascending=False)