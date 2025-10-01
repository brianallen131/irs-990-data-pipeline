import pandas as pd
import zipfile_deflate64
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from threading import Lock
import concurrent.futures

from pathlib import Path

project_root = Path().parent
project_root


from pathlib import Path

grant_data_dir = Path("data/extracted_irs_data/contributions_grants_assets")
grant_files = list(grant_data_dir.glob('*.csv'))


dataframes = []
for grant_file in grant_files:
    dataframes.append(pd.read_csv(grant_file))

grant_data = pd.concat(dataframes)

grant_data.columns

grant_data_990ez = grant_data[grant_data['form']=='990EZ'].sort_values(by=['grant_amt'],ascending=False)

grant_data_990pf = grant_data[grant_data['form']=='990PF'].sort_values(by=['grant_amt'],ascending=False)

grant_data_990 = grant_data[grant_data['form']=='990'].sort_values(by=['daf_managed_assets_at_eoy'],ascending=False)


grant_data.groupby(['form'])['ein'].count().sort_values(ascending=False)




def format_currency_with_abbreviations(number):
    """
    Formats a number as currency with commas, dollar sign, and M, B, T abbreviations.
    """
    if abs(number) >= 1_000_000_000_000:  # Trillions
        return f"${number / 1_000_000_000_000:,.2f}T"
    elif abs(number) >= 1_000_000_000:  # Billions
        return f"${number / 1_000_000_000:,.2f}B"
    elif abs(number) >= 1_000_000:  # Millions
        return f"${number / 1_000_000:,.2f}M"
    else:
        return f"${number:,.2f}"

year_sum['daf_managed_assets_at_eoy'].apply(format_currency_with_abbreviations)


GRANT_DETAILS_FIELDS




#######################

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

tmp = df_cga[df_cga['ein']==110303001]




#######################

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

df_daf.sort_values(by=['daf_managed_assets_at_eoy'],ascending=False,inplace=True)
df_daf_110303001 = df_daf[df_daf['ein']==110303001]




#######################

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
df_icd.columns

tmp = df_icd.groupby(['BusinessNameLine1Txt'])['CompensationAmt'].sum().sort_values(ascending=False)


index_2024= pd.read_csv('data/raw_irs_data/index_2024.csv')
tmp = index_2024[index_2024['EIN']==110303001]

with

import zipfile
with zipfile.ZipFile('data/raw_irs_data/2025_TEOS_XML_05A.zip', 'r') as zf:
    # Name of the file to be created to check if it exists, store for later for saving purposes

    # Get list of XML files within each zip file
    xml_files = [name for name in zf.namelist() if name.endswith('.xml')]
    print(xml_files)

xml_files_2025 = pd.DataFrame(xml_files)

xml_files_2025[xml_files_2025[0]=='202521349349310887_public.xml']
xml_files_2025[xml_files_2025[0]=='202501349339302145_public.xml']

import numpy as np
index_2025[index_2025['RETURN_ID'].isnull()]