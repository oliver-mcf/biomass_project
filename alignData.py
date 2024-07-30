
# Align Model Input Data

# Libraries ########################################################################################################
from libraries import *


# Objects & Methods ################################################################################################
def combine(csv_list, fixed_columns, output_csv):
    '''Merge initial intersecting data noting source file'''
    combined_df = pd.DataFrame(columns = ['Source'] + fixed_columns)
    for csv in csv_list:
        df = pd.read_csv(csv)
        df['Source'] = os.path.basename(csv)[:6]
        column_mapping = {}
        for col in df.columns:
            for fixed_col in fixed_columns:
                if col.endswith(fixed_col):
                    column_mapping[fixed_col] = col
                    break
        for fixed_col in fixed_columns:
            if fixed_col not in column_mapping:
                df[fixed_col] = pd.NA
            else:
                df[fixed_col] = df[column_mapping[fixed_col]]
        df = df[['Source'] + fixed_columns]
        combined_df = pd.concat([combined_df, df], ignore_index = True)
    combined_df = combined_df.sort_values(by = 'Source')
    combined_df.to_csv(output_csv, index = False)
    print(combined_df.head())
    print(combined_df.shape)
    print(f'Successful merge of model input data into {output_csv}')

def decibels(dn):
    '''Calculate decibels from digital numbers'''
    return 10 * np.log10(dn ** 2) - 83.0    # Rosenqvist et al, 2007

def convert_dB(file):
    '''Convert all palsar variables to decibels'''
    df = pd.read_csv(file)
    columns_to_correct = ['HH_Median', 'HH_StDev', 'HH_95', 'HH_05', 
                          'HV_Median', 'HV_StDev', 'HV_95', 'HV_05']
    for col in columns_to_correct:
        if col in df.columns:
            df[col] = df[col].apply(decibels)
    df.to_csv(file, index = False)
    print(f'Successful conversion of PALSAR data from DN to dB')

def geo_palsar(input_csv, output_csv):
    '''Geolocation alignment with PALSAR'''
    # Read csv
    df = pd.read_csv(input_csv)
    # Calculate absolute difference
    df['AGB-HV'] = abs(df['GEDI_AGB'] - df['HV_Median'])
    # Mask differences less than 1 sigma
    std_diff = df['AGB-HV'].std()
    df['Geolocated'] = (df['AGB-HV'] < std_diff)
    # Filter based on geolocation accuracy
    filtered_df = df[df['Geolocated']]
    print("Filter(B):", len(filtered_df))
    # Save filtered dataframe
    filtered_df = filtered_df.drop(columns = ['AGB-HV', 'Geolocated'])
    filtered_df.to_csv(output_csv, index = False)

def geo_cover(input_csv, output_csv):
    '''Geolocation alignment with GEDI Cover'''
    # Read csv
    df = pd.read_csv(input_csv)
    # Calculate absolute difference
    df['COVER-NDVI'] = abs(df['GEDI_COVER'] - df['YR_NDVI_Median'])
    # Mask differences less than 1 sigma
    std_diff = df['COVER-NDVI'].std()
    df['Geolocated'] = (df['COVER-NDVI'] < std_diff)
    # Filter based on geolocation accuracy
    filtered_df = df[df['Geolocated']]
    print("Filter(C):", len(filtered_df))
    # Save filtered dataframe
    filtered_df = filtered_df.drop(columns = ['COVER-NDVI', 'Geolocated'])
    filtered_df.to_csv(output_csv, index = False)


# Code #############################################################################################################
if __name__ == '__main__':
    
    # Define command line arguments
    parser = argparse.ArgumentParser(description = 'Combine CSV files for model training')
    parser.add_argument('--site', help = 'Site name to filter CSV files')
    parser.add_argument('--geo', help = 'Either "PALSAR" or "COVER" geolocation filter')
    args = parser.parse_args()

    # Identify filtered input data
    csv_list = glob(f'/home/s1949330/data/diss_data/pred_vars/input_init/*.csv')
    
    # Define variable names
    fixed_columns = ['GEDI_X', 'GEDI_Y', 'GEDI_AGB', 'GEDI_COVER',
                     'SR_B2_Median', 'SR_B2_StDev', 'SR_B2_p95', 'SR_B2_p05', 
                     'SR_B3_Median', 'SR_B3_StDev', 'SR_B3_p95', 'SR_B3_p05',
                     'SR_B4_Median', 'SR_B4_StDev', 'SR_B4_p95', 'SR_B4_p05',
                     'SR_B5_Median', 'SR_B5_StDev', 'SR_B5_p95', 'SR_B5_p05',
                     'SR_B6_Median', 'SR_B6_StDev', 'SR_B6_p95', 'SR_B6_p05',
                     'SR_B7_Median', 'SR_B7_StDev', 'SR_B7_p95', 'SR_B7_p05',
                     'YR_NDVI_Median', 'T1_NDVI_Median', 'T2_NDVI_Median', 'T3_NDVI_Median',
                     'NDVI_Wet95', 'NDVI_Wet05', 'NDVI_Dry95', 'NDVI_Dry05', 'NDVI_Gradient',
                     'SRTM_Elevation', 'SRTM_Slope', 'SRTM_mTPI',
                     'VV_Median', 'VV_StDev', 'VV_95', 'VV_05',
                     'VH_Median', 'VH_StDev', 'VH_95', 'VH_05',
                     'HH_Median', 'HH_StDev', 'HH_95', 'HH_05',
                     'HV_Median', 'HV_StDev', 'HV_95', 'HV_05',
                     'HHHV_Ratio', 'HHHV_Index']
    
    # Output file with site condition
    if args.site:
        csv_list = [csv for csv in csv_list if os.path.basename(csv).startswith(args.site)]
        merge_csv = f'/home/s1949330/data/diss_data/pred_vars/input_merge/All_{args.site}_EXTRACT_MERGE.csv'
    else:
        merge_csv = f'/home/s1949330/data/diss_data/pred_vars/input_merge/All_EXTRACT_MERGE.csv'
    
    # Align and export combined model input data
    combine(csv_list, fixed_columns, merge_csv)

    # Convert Palsar data to decibels
    convert_dB(merge_csv)

    # Filter data by geolocation condition
    output_csv = merge_csv.replace('_MERGE.csv', f'_MERGE_{args.geo}.csv')
    if args.geo == 'PALSAR':
        geo_palsar(merge_csv, output_csv)
    elif args.geo == 'COVER':
        geo_cover(merge_csv, output_csv)
