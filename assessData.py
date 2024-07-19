
# Assess Available Training Data by Year and Site


# Libraries ########################################################################################################
from libraries import *


# Objects & Methods ################################################################################################
def sample_year(file):       
    df = pd.read_csv(file)
    print(len(df))
    # Comment next three lines to access data without no data filtering
    df = df.dropna()
    df = df[(df != 0).all(axis = 1)]
    print(len(df))
    source_counts = df['Source'].value_counts()
    for source, count in sorted(source_counts.items()):
        print(f"{source}: {count}")

def var_regress(file, geo):
    '''Perform linear regression with target and predictor variables'''
    # Read file
    df = pd.read_csv(file)
    df = df.dropna()
    df = df[(df != 0).all(axis = 1)]
    # Isolate target and predictor variables
    y = df['GEDI_AGB']    
    pred_vars = df.drop(columns = ['Source', 'GEDI_X', 'GEDI_Y', 'GEDI_COVER', 'GEDI_AGB'])
    # Prepare a dictionary to hold regression results
    results = {
        'Variable': [],
        'Coefficient': [],
        'Intercept': [],
        'R2 Score': []}
    # Perform linear regression for each predictor variable
    for var in pred_vars.columns:
        X_column = pred_vars[[var]]
        model = LinearRegression()
        model.fit(X_column, y)
        y_pred = model.predict(X_column)
        r2 = r2_score(y, y_pred)
        results['Variable'].append(var)
        results['Coefficient'].append(model.coef_[0])
        results['Intercept'].append(model.intercept_)
        results['R2 Score'].append(r2)
    # Convert results dictionary to a DataFrame
    results_df = pd.DataFrame(results)
    if geo:
        output_csv = f'/home/s1949330/data/diss_data/pred_vars/input_merge/All_LINEAR_REGRESSION_{geo}.csv'
    else:
        output_csv = '/home/s1949330/data/diss_data/pred_vars/input_merge/All_LINEAR_REGRESSION.csv'
    results_df.to_csv(output_csv, index = False)
    print(f'SUCCESS: Regression results saved to {output_csv}')




# Code #############################################################################################################
if __name__ == '__main__':

    # Determine command line arguments
    parser = argparse.ArgumentParser(description = 'Assess data stored in csv for filtered training data')
    parser.add_argument('--count', action = 'store_true', help = 'Boolean to count footprints/intersects in file')
    parser.add_argument('--file', help = 'Filename of csv with directory, to return n footprints by unique value (site_year)')
    parser.add_argument('--geo', help = 'Geolocation filter condition, PALSAR or COVER')
    parser.add_argument('--regress', action = 'store_true', help = 'Instruction to perform linear regression with predictor variables')
    args = parser.parse_args()
    
    if args.count:
        #file = '/home/s1949330/data/diss_data/pred_vars/input_final/All_MODEL_INPUT_GEO_FINAL.csv'
        sample_year(args.file)

    if args.regress:
        var_regress(args.file, args.geo)
