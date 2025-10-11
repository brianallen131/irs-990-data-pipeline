from pathlib import Path
import pandas as pd
import re
from rapidfuzz import fuzz, process
from multiprocessing import Pool, cpu_count


def clean_name(name):
    """Standardize organization names"""
    if pd.isna(name):
        return ''

    name = str(name).upper().strip()
    name = re.sub(r'[^\w\s]', ' ', name)

    # Standardize common suffixes
    replacements = {
        r'\bINCORPORATED\b': 'INC',
        r'\bINCORP\b': 'INC',
        r'\bCORPORATION\b': 'CORP',
        r'\bCOMPANY\b': 'CO',
        r'\bFOUNDATION\b': 'FDN',
        r'\bFOUND\b': 'FDN',
        r'\bASSOCIATION\b': 'ASSN',
        r'\bASSOC\b': 'ASSN',
        r'\bCHARITABLE\b': 'CHAR',
        r'\bUNIVERSITY\b': 'UNIV',
        r'\bCOLLEGE\b': 'COLL',
        r'\bNATIONAL\b': 'NATL',
        r'\bINTERNATIONAL\b': 'INTL',
        r'\bLIMITED\b': 'LTD',
        r'\bSOCIETY\b': 'SOC',
        r'\bCENTER\b': 'CTR',
        r'\bCENTRE\b': 'CTR',
        r'\bDEPARTMENT\b': 'DEPT',
        r'\bTRUST\b': 'TR',
        r'\bAND\b': '&',
    }

    for pattern, replacement in replacements.items():
        name = re.sub(pattern, replacement, name)

    name = re.sub(r'\b(INC|LLC|CORP|LTD|CO)\b', '', name)
    name = re.sub(r'\s+', ' ', name).strip()

    return name


def clean_city(city):
    """Standardize city names"""
    if pd.isna(city):
        return ''

    city = str(city).upper().strip()
    replacements = {
        r'\bSAINT\b': 'ST',
        r'\bFORT\b': 'FT',
        r'\bMOUNT\b': 'MT',
    }

    for pattern, replacement in replacements.items():
        city = re.sub(pattern, replacement, city)

    city = re.sub(r'[^\w\s]', ' ', city)
    city = re.sub(r'\s+', ' ', city).strip()

    return city


def clean_state(state):
    """Standardize state codes"""
    if pd.isna(state):
        return ''
    return str(state).upper().strip()[:2]


def clean_zip(zip_code):
    """Standardize ZIP codes to 5 digits"""
    if pd.isna(zip_code):
        return ''

    zip_str = str(zip_code).strip()
    zip_match = re.search(r'\d{5}', zip_str)
    return zip_match.group(0) if zip_match else ''


def match_state_chunk(args):
    """Match grants for a single state (runs in parallel)"""
    state, unique_grants, df_bmf, threshold = args

    state_grants = unique_grants[unique_grants['clean_recipient_state'] == state].copy()
    state_bmf = df_bmf[df_bmf['clean_state'] == state].copy()

    if len(state_bmf) == 0:
        print(f"State {state}: No BMF records, skipping {len(state_grants):,} grants")
        return []

    print(f"State {state}: Matching {len(state_grants):,} recipients against {len(state_bmf):,} orgs")

    # Create lookup
    state_bmf['lookup_key'] = state_bmf['clean_name'] + '|' + state_bmf['clean_city']
    bmf_lookup = dict(zip(state_bmf['lookup_key'], state_bmf['EIN']))
    bmf_keys = list(bmf_lookup.keys())

    results = []

    for idx, row in state_grants.iterrows():
        query = row['clean_recipient_name'] + '|' + row['clean_recipient_city']

        # Try exact ZIP match first
        if row['clean_recipient_zip']:
            zip_match = state_bmf[state_bmf['clean_zip'] == row['clean_recipient_zip']]
            if len(zip_match) > 0:
                zip_keys = (zip_match['clean_name'] + '|' + zip_match['clean_city']).tolist()
                match = process.extractOne(query, zip_keys, scorer=fuzz.token_sort_ratio, score_cutoff=threshold)
                if match:
                    matched_key, score, _ = match
                    ein = bmf_lookup[matched_key]
                    results.append({'index': idx, 'EIN': ein, 'match_score': score, 'match_type': 'zip+fuzzy'})
                    continue

        # Fallback to state-wide fuzzy match
        match = process.extractOne(query, bmf_keys, scorer=fuzz.token_sort_ratio, score_cutoff=threshold)
        if match:
            matched_key, score, _ = match
            ein = bmf_lookup[matched_key]
            results.append({'index': idx, 'EIN': ein, 'match_score': score, 'match_type': 'state_fuzzy'})
        else:
            results.append({'index': idx, 'EIN': None, 'match_score': 0, 'match_type': 'no_match'})

    print(f"State {state}: Complete")
    return results


def match_grants_to_bmf(df_grants, df_bmf, threshold=85, n_jobs=None):
    """Match grants to BMF organizations with parallel processing"""

    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)

    print("Step 1: Cleaning grants data...")
    grants = df_grants.copy()
    grants['clean_recipient_name'] = grants['recipient_name'].apply(clean_name)
    grants['clean_recipient_city'] = grants['recipient_city'].apply(clean_city)
    grants['clean_recipient_state'] = grants['recipient_state'].apply(clean_state)
    grants['clean_recipient_zip'] = grants['recipient_zip'].apply(clean_zip)

    # Remove rows with missing critical fields
    grants = grants[grants['clean_recipient_name'].str.len() > 0]
    grants = grants[grants['clean_recipient_state'].str.len() > 0]
    print(f"After cleaning: {len(grants):,} grants")

    print("\nStep 2: Aggregating to unique recipients...")
    unique_grants = grants.groupby(
        ['clean_recipient_name', 'clean_recipient_city', 'clean_recipient_state', 'clean_recipient_zip'],
        as_index=False
    ).agg({
        'recipient_name': 'first',
        'recipient_city': 'first',
        'recipient_state': 'first',
        'recipient_zip': 'first',
    })
    print(f"Unique recipients: {len(unique_grants):,}")

    print("\nStep 3: Cleaning BMF data...")
    bmf = df_bmf.copy()
    bmf['clean_name'] = bmf['NAME'].apply(clean_name)
    bmf['clean_city'] = bmf['CITY'].apply(clean_city)
    bmf['clean_state'] = bmf['STATE'].apply(clean_state)
    bmf['clean_zip'] = bmf['ZIP'].apply(clean_zip)
    print(f"BMF records: {len(bmf):,}")

    print(f"\nStep 4: Matching by state using {n_jobs} parallel workers...")
    states = unique_grants['clean_recipient_state'].unique().tolist()
    print(f"Processing {len(states)} states in parallel...\n")

    # Create argument tuples for each state
    args_list = [(state, unique_grants, bmf, threshold) for state in states]

    # Process states in parallel
    with Pool(n_jobs) as pool:
        state_results = pool.map(match_state_chunk, args_list)

    # Flatten results
    all_results = [item for sublist in state_results for item in sublist]

    print("\nStep 5: Merging results...")
    if all_results:
        results_df = pd.DataFrame(all_results).set_index('index')
        unique_grants = unique_grants.join(results_df[['EIN', 'match_score', 'match_type']])
    else:
        unique_grants['EIN'] = None
        unique_grants['match_score'] = 0
        unique_grants['match_type'] = 'no_match'

    # Merge back to original grants
    grants = grants.merge(
        unique_grants[
            ['clean_recipient_name', 'clean_recipient_city', 'clean_recipient_state', 'clean_recipient_zip', 'EIN',
             'match_score', 'match_type']],
        on=['clean_recipient_name', 'clean_recipient_city', 'clean_recipient_state', 'clean_recipient_zip'],
        how='left'
    )

    matched = grants['EIN'].notna().sum()
    print(f"\n✓ Matched {matched:,} of {len(grants):,} grants ({matched / len(grants) * 100:.1f}%)")
    print(f"\nMatch breakdown:")
    print(grants['match_type'].value_counts())

    return grants


if __name__ == '__main__':
    print("Loading data...")

    df_grants = pd.read_parquet('data/compiled_irs_data/grant_details_compiled.parquet')
    bmf_dir = Path('data/downloaded_irs_data')
    dataframes = []
    for file in list(bmf_dir.glob('*.csv')):
        dataframes.append(pd.read_csv(file))
    df_bmf = pd.concat(dataframes)

    print(f"Initial grants: {len(df_grants):,}")
    print(f"BMF records: {len(df_bmf):,}\n")

    matched_grants = match_grants_to_bmf(df_grants, df_bmf, threshold=85, n_jobs=None)

    print("\nSaving results...")
    matched_grants.to_parquet('data/cleaned_irs_data/grant_details_recipients_inferred.parquet', engine='fastparquet')
    print("Done!")
