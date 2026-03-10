import pandas as pd
from pathlib import Path
import argparse

# Updated data roots
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT /  "Preprocessing" / "raw_data"
OUTPUT_ROOT = ROOT

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
                          OUTPUT_ROOT:str,
                          OUTPUT_FILE_NAME: str,
                          filepath_860: str, filepath_923: str,
                          filepath_plant: str,
                          capacity_factor_threshold: float = 0.0,
                          nerc_filter: str = None) -> dict:
    """
    Methodology: Sum annual MWH from 923, divide by 8760 * nameplate capacity
    to get capacity factor. If capacity factor > threshold, include in output dict.
    Returns dict keyed by (plant_code, generator_id) with fuel, nameplate, CF.
    """

    # Load EIA 860 for nameplate capacity
    filepath_860 = ROOT / filepath_860
    df_860 = pd.read_csv(filepath_860, header=1)
    df_860.columns = df_860.columns.str.strip()

    # Keep relevant columns
    cols_860 = {
                'Plant Code':              'plant_code',
                'Plant Name':              'plant_name',
                'Technology':              'technology',
                'Nameplate Capacity (MW)': 'nameplate_mw',
                'Status':                  'status',
                }
    df_860 = df_860[[c for c in cols_860 if c in df_860.columns]].rename(columns=cols_860)
    df_860['technology'] = df_860['technology'].map(map_technology)
    unmapped = df_860[~df_860['technology'].isin(TECHNOLOGY_MAP.values())]['technology'].unique()
    if len(unmapped):
        print(f"Unmapped technologies: {unmapped}")
    df_860['plant_code'] = pd.to_numeric(df_860['plant_code'], errors='coerce')

    # Load plant file to grab relevant ISO/NERC region
    filepath_plant = ROOT / filepath_plant
    df_plant = pd.read_csv(filepath_plant, header=1)
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
    df_860['plant_code'] = pd.to_numeric(df_860['plant_code'], errors='coerce')

    # Load EIA-923 for annual generation parsing
    filepath_923 = ROOT / filepath_923
    df_923 = pd.read_csv(filepath_923, header=5)
    df_923.columns = df_923.columns.str.strip()
    # Grab relevant columns
    annual_col = next(c for c in df_923.columns if 'Net Generation' in str(c) and 'Megawatt' in str(c))
    plant_col = next((c for c in df_923.columns if 'Plant Id' in c or 'Plant Code' in c), None)
    # Dataframe filtering + datatype conversion
    df_923 = df_923[[plant_col, annual_col]].copy()
    df_923 = df_923.rename(columns={plant_col: 'plant_code', annual_col: 'annual_mwh'})
    df_923['plant_code'] = pd.to_numeric(df_923['plant_code'], errors='coerce')
    df_923['annual_mwh'] = pd.to_numeric(df_923['annual_mwh'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    df_923_agg = df_923.groupby('plant_code')['annual_mwh'].sum().reset_index()
    df_923['plant_code'] = pd.to_numeric(df_923['plant_code'], errors='coerce')
    # Merge on plant code + filter for active generators
    merged = df_860.merge(df_923_agg, on='plant_code', how='left')
    merged = merged.drop(columns = "nameplate_mw")
    merged['annual_mwh'] = merged['annual_mwh'].fillna(0)
    merged = merged.drop_duplicates()

    merged['capacity_factor'] = merged['annual_mwh'] / (merged['nameplate_mw_total'] * 8760)
    merged = merged[merged['capacity_factor'] >= capacity_factor_threshold]

    # --- Build output dictionary ---
    output_dict = {}
    for _, row in merged.iterrows():
        # Key - Technology + Plantcode
        key = str(row['technology']) + "_" + str(int(row['plant_code']))
        # Comment out some results
        output_dict[key] = {
            'plant_name':         row['plant_name'],
            # 'plant_code':         row['plant_code'],  
            'fuel_type':         row['technology'],
            # 'nerc_region':        row['nerc_region'],
            'capacity': row['nameplate_mw_total'],
            # 'annual_mwh':         round(row['annual_mwh'], 2),
            'capacity_factor_2025':    round(row['capacity_factor'], 4),
        }
    # Output file
    output_file_name = OUTPUT_ROOT / OUTPUT_FILE_NAME
    pd.DataFrame.from_dict(output_dict, orient='index').rename_axis('Power_Plant_ID').to_csv(output_file_name)
    return output_dict


## Grab fuel mix and break down by hour + convert to percent
def load_fuel_mix_raw(ROOT: str,
                        OUTPUT_ROOT:str,
                        OUTPUT_FILE_NAME: str,
                        data_filepaths: list) -> pd.DataFrame:
    sheet_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    all_dfs = []
    
    for filepath in data_filepaths:
        filepath = ROOT / filepath
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
    # Export combined csv for raw data
    raw_fuel_mix_file_name = OUTPUT_ROOT / OUTPUT_FILE_NAME 
    combined.to_csv(raw_fuel_mix_file_name)

    return combined

def preprocess_fuel_mix(OUTPUT_ROOT: str,
                    OUTPUT_FILE_NAME: str,
                    combined: pd.DataFrame) -> pd.DataFrame:
    combined_pos = combined.clip(lower=0)
    total = combined_pos.sum(axis=1)
    pct = combined_pos.div(total, axis=0)
    pct = pct.round(6)
    pct.insert(0, 'Total_MW', total)
    pct.to_csv(OUTPUT_ROOT / OUTPUT_FILE_NAME)
    return pct

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    p.add_argument("--load_fuel_mix_from_excel", action="store_true", help="If re-running of excel fuel mix data required")
    p.add_argument('--fuel_mix_filepaths', nargs='+',
                   default=["IntGenbyFuel2025.xlsx", "IntGenbyFuel2026.xlsx"])
    p.add_argument('--fuel_mix_raw_output',
                   default="IntGenbyFuel20252026.csv", help="Used if data exists")
    p.add_argument('--fuel_mix_pct_output',
                   default="fuel_mix_2025_2026_ercot.csv")
    p.add_argument('--filepath_860',   default='3_1_Generator_Y2024.csv')
    p.add_argument('--filepath_923',   default='EIA923_Schedules_2_3_4_5_M_12_2025_20FEB2026.csv')
    p.add_argument('--filepath_plant', default='2___Plant_Y2024.csv')
    p.add_argument('--generators_output', default='active_generators_2025_ercot.csv')
    p.add_argument('--capacity_factor_threshold', type=float, default=0.05)
    p.add_argument('--nerc_filter', default='TRE', help = "To denote the balancing authority data used")
    return p.parse_args()
if __name__ == "__main__":
    # Ex. Command: python Preprocessing/preprocessing.py --load_fuel_mix_from_excel --verbose
    args = parse_args()
    if args.load_fuel_mix_from_excel:
        fuel_mix_merged_csv = load_fuel_mix_raw(
            ROOT=DATA_ROOT / "xlsx",
            OUTPUT_ROOT=DATA_ROOT / "csv",
            OUTPUT_FILE_NAME=args.fuel_mix_raw_output,
            data_filepaths=args.fuel_mix_filepaths
        )
    else:
        fuel_mix_merged_csv = pd.read_csv(DATA_ROOT / "csv" / args.fuel_mix_raw_output, index_col=0, parse_dates=True)
        fuel_mix_merged_csv = fuel_mix_merged_csv.astype(float)

    result = preprocess_fuel_mix(
        OUTPUT_ROOT=OUTPUT_ROOT,
        OUTPUT_FILE_NAME=args.fuel_mix_pct_output,
        combined=fuel_mix_merged_csv
    )

    generators = get_active_generators(
        ROOT=DATA_ROOT / "csv",
        OUTPUT_ROOT=OUTPUT_ROOT,
        OUTPUT_FILE_NAME=args.generators_output,
        filepath_860=args.filepath_860,
        filepath_923=args.filepath_923,
        filepath_plant=args.filepath_plant,
        capacity_factor_threshold=args.capacity_factor_threshold,
        nerc_filter=args.nerc_filter
    )

    if args.verbose:
        capacity = 0.0
        capacity_by_fuel = {}
        for key in generators.keys():
            row = generators[key]
            capacity += row['capacity']
            fuel = row['fuel_type']
            capacity_by_fuel[fuel] = capacity_by_fuel.get(fuel, 0.0) + row['capacity']
        print(f"Total capacity: {capacity} MW")
        for fuel, cap in sorted(capacity_by_fuel.items()):
            print(f"  {fuel}: {cap} MW")
