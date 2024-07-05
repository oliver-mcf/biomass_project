
# Filter Predictor Variables / Model Features

# Libraries ########################################################################################################
from libraries import *


# Objects & Methods ################################################################################################
def matrix(df, coef):
    '''Create correlation matrix for predictor variables and filter highly correlated ones.'''
    original_columns = ['Source', 'GEDI_X', 'GEDI_Y', 'GEDI_AGB']
    predictor_columns = df.columns[4:]
    # Compute the correlation matrix
    correlation_matrix = df[predictor_columns].corr()
    # Plot and save the correlation matrix heatmap
    plt.rcParams['font.family'] = 'Arial'
    plt.figure(figsize = (24, 20))
    sns.heatmap(correlation_matrix, annot = True, fmt = ".1f", cmap = 'coolwarm', square = True,
                cbar_kws = {"shrink": .5}, mask = np.eye(len(correlation_matrix), dtype = bool),
                vmin = -1, vmax = 1)
    plt.title('Predictor Variable Correlation Matrix')
    plt.savefig('/home/s1949330/Documents/scratch/diss_data/pred_vars/CORRELATION_MATRIX.png', dpi = 300)
    plt.close()
    # Identify and remove highly correlated variables
    variables_to_keep = set(predictor_columns)
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) >= coef:
                if correlation_matrix.columns[j] in variables_to_keep:
                    variables_to_keep.remove(correlation_matrix.columns[j])
    
    print("Variables to be removed:")
    variables_to_remove = set(predictor_columns) - variables_to_keep
    print("Number of variables removed:", len(variables_to_remove))
    pprint(variables_to_remove)
    print("Variables to be kept:")
    print("Number of variables kept:", len(variables_to_keep))
    pprint(variables_to_keep)
    # Save the filtered dataframe to a new CSV file
    variables_to_keep = original_columns + list(variables_to_keep)
    df_filtered = df[variables_to_keep]
    output_csv = '/home/s1949330/Documents/scratch/diss_data/pred_vars/input_final/MODEL_INPUT_FINAL.csv'
    df_filtered.to_csv(output_csv, index = False)
    print(f"Filtered data saved to: {output_csv}")

def reduce(ref_csv, csv_list, output_dir):
    '''Reduce predictor variables by established threshold for sites'''
    # Read already filtered csv
    ref_df = pd.read_csv(ref_csv)
    ref_columns = set(ref_df.columns)
    # Reduce predictor variables in site csvs
    for csv_file in csv_list:
        df = pd.read_csv(csv_file)
        common_columns = ref_columns.intersection(df.columns)
        filtered_df = df[common_columns]
        # Write newly filtered site input data to csv
        filename = os.path.basename(csv_file).replace('_MERGE.csv', '_FINAL.csv')
        output_path = os.path.join(output_dir, filename)
        filtered_df.to_csv(output_path, index = False)
        print(f"Site filtered data saved to: {output_path}")


# Code #############################################################################################################
if __name__ == '__main__':

    # Define command line arguments
    parser = argparse.ArgumentParser(description = 'Filtering Predictor Variables for Model Input')
    parser.add_argument('--filter', action = 'store_true', help = 'Filter predictor variables of main model input csv')
    parser.add_argument('--coef', type = float, help = 'Correlation coefficient threshold for filtering predictor variables')
    parser.add_argument('--reduce', action = 'store_true', help = 'Reduce predictor variables in site specific data based on previous filtering')
    args = parser.parse_args()

    # Filter predictor variables by correlation coefficients
    if args.filter:
        df = pd.read_csv('/home/s1949330/Documents/scratch/diss_data/pred_vars/input_merge/MODEL_INPUT_MERGE.csv')
        matrix(df, args.coef)

    # Match site specific subsets by filtered predictor variables
    if args.reduce:
        ref_csv = '/home/s1949330/Documents/scratch/diss_data/pred_vars/input_final/MODEL_INPUT_FINAL.csv'
        csv_list = ['/home/s1949330/Documents/scratch/diss_data/pred_vars/input_merge/MGR_MODEL_INPUT_MERGE.csv', 
                    '/home/s1949330/Documents/scratch/diss_data/pred_vars/input_merge/TKW_MODEL_INPUT_MERGE.csv']
        output_dir = '/home/s1949330/Documents/scratch/diss_data/pred_vars/input_final/'
        reduce(ref_csv, csv_list, output_dir)
