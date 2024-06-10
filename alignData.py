
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
    combined_df.to_csv(output_csv, index = False)
    print(combined_df.head())
    print(combined_df.shape)
    print(f'Successful merge of model input data into {output_csv}')


# Code #############################################################################################################
if __name__ == '__main__':
    
    csv_list = glob(f'/home/s1949330/Documents/scratch/diss_data/*.csv')
    fixed_columns = ['GEDI_X', 'GEDI_Y', 'GEDI_AGB',
                     'SR_B2_Median', 'SR_B2_StDev', 'SR_B2_p95', 'SR_B2_p05', 
                     'SR_B3_Median', 'SR_B3_StDev', 'SR_B3_p95', 'SR_B3_p05',
                     'SR_B4_Median', 'SR_B4_StDev', 'SR_B4_p95', 'SR_B4_P05',
                     'SR_B5_Median', 'SR_B5_StDev', 'SR_B5_p95', 'SR_B5_p05',
                     'SR_B6_Median', 'SR_B6_StDev', 'SR_B6_p95', 'SR_B6_p05',
                     'SR_B7_Median', 'SR_B7_StDev', 'SR_B7_p95', 'SR_B7_p05',
                     '08_NDVI_Median', '09_NDVI_Median', '10_NDVI_Median', '11_NDVI_Median', '12_NDVI_Median',
                     '01_NDVI_Median', '02_NDVI_Median', '03_NDVI_Median', '04_NDVI_Median', '05_NDVI_Median', '06_NDVI_Median', '07_NDVI_Median',
                     'NDVI_Wet95', 'NDVI_Wet05', 'NDVI_Dry95', 'NDVI_Dry05', 'NDVI_Gradient',
                     'SRTM_Elevation', 'SRTM_Slope', 'SRTM_mTPI']
    output_csv = '/home/s1949330/Documents/scratch/diss_data/model/MODEL_INPUT.csv'
    combine(csv_list, fixed_columns, output_csv)
    combined_df = pd.read_csv(output_csv)
    pprint(combined_df.columns.tolist())