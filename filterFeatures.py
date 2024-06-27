
# Filter Predictor Variables / Model Features

# Libraries ########################################################################################################
from libraries import *


# Objects & Methods ################################################################################################
def matrix(csv_file):
    # Create correlation matrix for predictor variables
    df = pd.read_csv(csv_file)
    original_columns = ['Source', 'GEDI_X', 'GEDI_Y', 'GEDI_AGB']
    predictor_columns = df.columns[4:].tolist()
    correlation_matrix = df[predictor_columns].corr()
    mask = np.eye(len(correlation_matrix), dtype = bool)
    plt.rcParams['font.family'] = 'Arial'
    plt.figure(figsize = (24, 20))
    # Visualise correlation matrix
    sns.heatmap(correlation_matrix, annot = True, fmt = ".1f", cmap = 'coolwarm', square = True, 
                cbar_kws = {"shrink": .5}, mask = mask, vmin = -1, vmax = 1)
    for i in range(len(correlation_matrix)):
        plt.gca().add_patch(plt.Rectangle((i, i), 1, 1, fill = True, color = 'black', lw = 0))
    plt.title('Correlation Matrix Heatmap')
    plt.savefig('/home/s1949330/Documents/scratch/diss_data/model/CORRELATION_MATRIX.png', dpi = 300)
    plt.show()
    # Isolate variables with no coefficients > 0.9
    variables_to_remove = set()
    variables_to_keep = set(correlation_matrix.columns)
    for i in range(len(correlation_matrix.columns)):
        found_high_corr = False
        for j in range(len(correlation_matrix.columns)):
            if i != j and abs(correlation_matrix.iloc[i, j]) > 0.9:
                found_high_corr = True
                break
        if not found_high_corr:
            variables_to_remove.add(correlation_matrix.columns[i])
    print("Variables to be removed:")
    pprint(variables_to_remove)
    variables_to_keep -= variables_to_remove
    variables_to_remove = sorted(list(variables_to_remove))
    variables_to_keep = sorted(list(variables_to_keep))
    # Filter original predictor variables
    variables_to_keep = original_columns + list(variables_to_keep)
    df_filtered = df[variables_to_keep]
    output_csv = '/home/s1949330/Documents/scratch/diss_data/model/MODEL_INPUT_FILTER.csv'
    df_filtered.to_csv(output_csv, index = False)
    print(f"Filtered data saved to: {output_csv}")

def reduce(ref_csv, csv_list):
    ref_df = pd.read_csv(ref_csv)
    ref_columns = set(ref_df.columns)
    for csv_file in csv_list:
        df = pd.read_csv(csv_file)
        common_columns = ref_columns.intersection(df.columns)
        filtered_df = df[common_columns]
        output = csv_file.replace('.csv', '_FILTER.csv')
        filtered_df.to_csv(output, index = False)
        print(f"Site filtered data saved to: {output}")


# Code #############################################################################################################
if __name__ == '__main__':

    # Define command line arguments
    parser = argparse.ArgumentParser(description = 'Filtering Predictor Variables for Model Input')
    parser.add_argument('--filter', action = 'store_true', help = 'Filter predictor variables of main model input csv')
    parser.add_argument('--reduce', action = 'store_true', help = 'Reduce predictor variables in site specific data based on previous filtering')
    args = parser.parse_args()

    # Filter predictor variables by correlation coefficients
    if args.filter:
        csv_file = '/home/s1949330/Documents/scratch/diss_data/model/MODEL_INPUT.csv'
        matrix(csv_file)

    # Match site specific subsets by filtered predictor variables
    if args.reduce:
        ref_csv = '/home/s1949330/Documents/scratch/diss_data/model/MODEL_INPUT_FILTER.csv'
        csv_list = ['/home/s1949330/Documents/scratch/diss_data/model/MGR_MODEL_INPUT.csv', 
                    '/home/s1949330/Documents/scratch/diss_data/model/MGR_MODEL_INPUT.csv']
        reduce(ref_csv, csv_list)
