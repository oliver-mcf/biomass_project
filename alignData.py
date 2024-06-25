
# Align Model Input Data

# Libraries ########################################################################################################
from libraries import *


# Objects & Methods ################################################################################################
def combine(csv_list, fixed_columns, output_csv):
    combined_df = pd.DataFrame(columns = ['Source'] + fixed_columns)
    for csv in csv_list:
        df = pd.read_csv(csv)
        df['Source'] = os.path.basename(csv)
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
    return 10 * np.log10(dn ** 2) - 83.0

def convert_palsar(file):
    df = pd.read_csv(file)
    columns_to_correct = ['HH_Median', 'HH_StDev', 'HH_95', 'HH_05', 
                          'HV_Median', 'HV_StDev', 'HV_95', 'HV_05']
    for col in columns_to_correct:
        if col in df.columns:
            df[col] = df[col].apply(decibels)
    df.to_csv(file, index = False)
    print(f'Successful conversion of PALSAR data from DN to dB')


# Code #############################################################################################################
if __name__ == '__main__':
    
    # Define command line arguments
    parser = argparse.ArgumentParser(description = 'Combine CSV files for model training')
    parser.add_argument('--site', help = 'Site name to filter CSV files')
    args = parser.parse_args()
    
    # Identify all model input data
    csv_list = glob(f'/home/s1949330/Documents/scratch/diss_data/model/csv/*.csv')

    if args.site:
        csv_list = [csv for csv in csv_list if os.path.basename(csv).startswith(args.site)]
        output_csv = f'/home/s1949330/Documents/scratch/diss_data/model/{args.site}_MODEL_INPUT.csv'
    else:
        output_csv = '/home/s1949330/Documents/scratch/diss_data/model/MODEL_INPUT.csv'
    
    fixed_columns = ['GEDI_X', 'GEDI_Y', 'GEDI_AGB',
                     'SR_B2_Median', 'SR_B2_StDev', 'SR_B2_p95', 'SR_B2_p05', 
                     'SR_B3_Median', 'SR_B3_StDev', 'SR_B3_p95', 'SR_B3_p05',
                     'SR_B4_Median', 'SR_B4_StDev', 'SR_B4_p95', 'SR_B4_p05',
                     'SR_B5_Median', 'SR_B5_StDev', 'SR_B5_p95', 'SR_B5_p05',
                     'SR_B6_Median', 'SR_B6_StDev', 'SR_B6_p95', 'SR_B6_p05',
                     'SR_B7_Median', 'SR_B7_StDev', 'SR_B7_p95', 'SR_B7_p05',
                     'T1_NDVI_Median', 'T2_NDVI_Median', 'T3_NDVI_Median',
                     'NDVI_Wet95', 'NDVI_Wet05', 'NDVI_Dry95', 'NDVI_Dry05', 'NDVI_Gradient',
                     'SRTM_Elevation', 'SRTM_Slope', 'SRTM_mTPI',
                     'VV_Median', 'VV_StDev', 'VV_95', 'VV_05',
                     'VH_Median', 'VH_StDev', 'VH_95', 'VH_05',
                     'HH_Median', 'HH_StDev', 'HH_95', 'HH_05',
                     'HV_Median', 'HV_StDev', 'HV_95', 'HV_05',
                     'HHHV_Ratio', 'HHHV_Index']
    
    # Align and export combined model input data
    combine(csv_list, fixed_columns, output_csv)

    # Convert Palsar data to decibels
    convert_palsar(output_csv)

