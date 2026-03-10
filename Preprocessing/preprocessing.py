# %%
import pandas as pd
import numpy as np
import os

FUEL_MIX_FILEPATHS = [r"IntGenbyFuel2025.xlsx", r"IntGenbyFuel2026.xlsx"]
ROOT = r"C:\Users\rrk23\OneDrive\Documents\GitHub\capstone-optimal-bidding\Preprocessing\raw_data"

## Mapper - technology -> fuel
TECHNOLOGY_MAP = {
    # Natural Gas
    'Natural Gas Fired Combined Cycle':        'Natural Gas',
    'Natural Gas Fired Combustion Turbine':    'Natural Gas',
    'Natural Gas Steam Turbine':               'Natural Gas',
    'Natural Gas Internal Combustion Engine':  'Natural Gas',
    'Other Natural Gas':                       'Natural Gas',
    # Coal
    'Conventional Steam Coal':                 'Coal',
    'Coal Integrated Gasification Combined Cycle': 'Coal',
    # Nuclear
    'Nuclear':                                 'Nuclear',
    # Solar
    'Solar Photovoltaic':                      'Solar',
    'Solar Thermal with Energy Storage':       'Solar',
    'Solar Thermal without Energy Storage':    'Solar',
    # Wind
    'Onshore Wind Turbine':                    'Wind',
    'Offshore Wind Turbine':                   'Wind',
    # Hydro
    'Conventional Hydroelectric':              'Hydro',
    'Run-of-River Hydroelectric':              'Hydro',
    'Pumped Storage':                          'Hydro',
    # Storage
    'Batteries':                               'Battery Storage',
    'Flywheels':                               'Battery Storage',
    # Other
    'Petroleum Liquids':                       'Oil',
    'Petroleum Coke':                          'Oil',
    'Landfill Gas':                            'Biomass',
    'Wood/Wood Waste Biomass':                 'Biomass',
    'Other Waste Biomass':                     'Biomass',
    'Geothermal':                              'Geothermal',
    'Natural Gas with Compressed Air Storage': 'Natural Gas',
    'Other Gases':                             'Other',
    'Hydrokinetic':                            'Other',
    'All Other':                               'Other',
}

def map_technology(tech: str) -> str:
    return TECHNOLOGY_MAP.get(tech, 'Other')

def get_active_generators(ROOT: str,
                          filepath_860: str, filepath_923: str, filepath_plant: str,
                          capacity_factor_threshold: float = 0.0,
                          nerc_filter: str = None) -> dict:
    """
    Methodology: Sum annual MWH from 923, divide by 8760 * nameplate capacity
    to get capacity factor. If capacity factor > threshold, include in output dict.
    Returns dict keyed by (plant_code, generator_id) with fuel, nameplate, CF.
    """

    # --- Load EIA-860 Operable sheet (header on row 2, data from row 3) ---
    filepath_860 = os.path.join(ROOT, filepath_860)
    df_860 = pd.read_excel(filepath_860, sheet_name='Operable', header=1)
    df_860.columns = df_860.columns.str.strip()

    # Keep relevant columns
    cols_860 = {
                'Plant Code':              'plant_code',
                'Plant Name':              'plant_name',
                'Technology':              'technology',
                'NERC Region':             'nerc_region',   # ADD
                'Nameplate Capacity (MW)': 'nameplate_mw',
                'Status':                  'status',
                }
    df_860 = df_860[[c for c in cols_860 if c in df_860.columns]].rename(columns=cols_860)
    df_860['technology'] = df_860['technology'].map(map_technology)
    unmapped = df_860[~df_860['technology'].isin(TECHNOLOGY_MAP.values())]['technology'].unique()
    if len(unmapped):
        print(f"Unmapped technologies: {unmapped}")

    # --- Load Plant file for NERC region ---
    df_plant = pd.read_excel(os.path.join(ROOT, filepath_plant), sheet_name='Plant', header=1)
    df_plant.columns = df_plant.columns.str.replace('\n', ' ').str.strip()
    df_plant = df_plant[['Plant Code', 'NERC Region']].rename(columns={'Plant Code': 'plant_code', 'NERC Region': 'nerc_region'})
    df_plant['plant_code'] = pd.to_numeric(df_plant['plant_code'], errors='coerce')
    if nerc_filter:
        df_plant = df_plant[df_plant['nerc_region'] == nerc_filter.upper()]


    # Filter to operating units only (status == 'OP') + optional state filter
    df_860 = df_860[df_860['status'] == 'OP']
    df_860 = df_860.merge(df_plant, on='plant_code', how='inner')

    df_860['plant_code'] = pd.to_numeric(df_860['plant_code'], errors='coerce')
    df_860['nameplate_mw'] = pd.to_numeric(df_860['nameplate_mw'], errors='coerce')
    df_860 = df_860.dropna(subset=['plant_code', 'nameplate_mw'])

    df_860_nameplate = df_860.groupby('plant_code')['nameplate_mw'].sum().reset_index()
    df_860 = df_860.merge(df_860_nameplate.rename(columns={'nameplate_mw': 'nameplate_mw_total'}), on='plant_code', how='left')

    # --- Load EIA-923 (header on row 6, i.e. header=5) ---
    filepath_923 = os.path.join(ROOT, filepath_923)
    df_923 = pd.read_excel(filepath_923, sheet_name='Page 1 Generation and Fuel Data', header=5)
    df_923.columns = df_923.columns.str.strip()

    netgen_cols = [c for c in df_923.columns if 'Netgen' in str(c) or 'Net Generation' in str(c)]
    plant_col = next((c for c in df_923.columns if 'Plant Id' in c or 'Plant Code' in c), None)

    df_923 = df_923[[plant_col] + netgen_cols].copy()
    df_923 = df_923.rename(columns={plant_col: 'plant_code'})
    df_923['plant_code'] = pd.to_numeric(df_923['plant_code'], errors='coerce')
    df_923[netgen_cols] = df_923[netgen_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Sum all monthly netgen columns -> annual MWh per plant
    df_923['annual_mwh'] = df_923[netgen_cols].sum(axis=1)
    df_923_agg = df_923.groupby('plant_code')['annual_mwh'].sum().reset_index()

    # --- Merge 860 + 923 on plant_code ---
    merged = df_860.merge(df_923_agg, on='plant_code', how='left')
    merged['annual_mwh'] = merged['annual_mwh'].fillna(0)
    df_860 = df_860.drop_duplicates(subset=['plant_code'])

    # Then use nameplate_mw_total in the re-calculate on entire plant:
    merged['capacity_factor'] = merged['annual_mwh'] / (merged['nameplate_mw_total'] * 8760)
    merged['capacity_factor'] = merged['capacity_factor'].clip(lower=0)  # no negatives

    merged = merged[merged['capacity_factor'] >= capacity_factor_threshold]

    # --- Build output dictionary ---
    result = {}
    for _, row in merged.iterrows():
        key = int(row['plant_code'])
        # Comment out some results
        result[key] = {
            'plant_name':         row['plant_name'],
            'fuel_type':         row['technology'],
            # 'nerc_region':        row['nerc_region'],
            'capacity': row['nameplate_mw_total'],
            # 'annual_mwh':         round(row['annual_mwh'], 2),
            # 'capacity_factor':    round(row['capacity_factor'], 4),
        }
    pd.DataFrame.from_dict(result, orient='index').to_csv(os.path.join(r"C:\Users\rrk23\OneDrive\Documents\GitHub\capstone-optimal-bidding", 'active_generators.csv'))
    return result


## Grab fuel mix and break down by hour + convert to percent
def preprocess_fuel_mix(ROOT: str, filepaths: list) -> pd.DataFrame:
    sheet_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    all_dfs = []
    
    for filepath in filepaths:
        filepath = os.path.join(ROOT, filepath)
        for sheet in sheet_names:
            try:
                df = pd.read_excel(filepath, sheet_name=sheet, header=0)
            except Exception:
                continue
            
            if df.empty:
                continue
            df = df.drop(columns=[col for col in df.columns if 'settlement' in str(col).lower()])

            id_vars = ['Date', 'Fuel']
            interval_cols = [c for c in df.columns if c not in id_vars + ['Total']]

            df = df.melt(id_vars=id_vars, value_vars=interval_cols,
                         var_name='interval', value_name='MW')

            df['interval'] = df['interval'].astype(str).str.strip()
            df['timedelta'] = pd.to_timedelta(df['interval'] + ':00')
            df['datetime'] = pd.to_datetime(df['Date']) + df['timedelta']
            df = df.drop(columns=['Date', 'interval', 'timedelta'])

            df = df.pivot_table(index='datetime', columns='Fuel', values='MW', aggfunc='sum')
            df.columns.name = None

            all_dfs.append(df)

    combined = pd.concat(all_dfs).sort_index()
    combined = combined.resample('h').sum()

    combined_pos = combined.clip(lower=0)
    total = combined_pos.sum(axis=1)
    pct = combined_pos.div(total, axis=0) * 100
    pct = pct.round(4)
    pct.insert(0, 'Total_MW', total)

    pct.to_csv(r"C:\Users\rrk23\OneDrive\Documents\GitHub\capstone-optimal-bidding\\fuel_mix_2025.csv")

    return pct

if __name__ == "__main__":
    result = preprocess_fuel_mix(ROOT=ROOT, filepaths=FUEL_MIX_FILEPATHS)
    # All ERCOT states (TX), CF > 5%
    generators = get_active_generators(ROOT=ROOT,
        filepath_860='3_1_Generator_Y2024.xlsx',
        filepath_923='EIA923_Schedules_2_3_4_5_M_12_2025_20FEB2026.xlsx',
        filepath_plant='2___Plant_Y2024.xlsx',
        capacity_factor_threshold=0.05,
        nerc_filter='TRE'
    )
    
    # Inspect one entry
    key = list(generators.keys())[0]
    print(key, generators[key])

    # Inspect sum
    result = 0.0
    for key in generators.keys():
        row = generators[key]
        result += row['capacity']
    print(f"Result: {result}")

# %%
