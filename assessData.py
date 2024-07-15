
# Assess Available Training Data by Year and Site


# Libraries ########################################################################################################
from libraries import *


# Objects & Methods ################################################################################################
def sample_year(file):       
    df = pd.read_csv(file)
    df = df.dropna()
    source_counts = df['Source'].value_counts()
    for source, count in source_counts.items():
        print(f"{source}: {count}")

def var_regress(file):
    '''Perform linear regression with target and predictor variables'''
    # Read file
    df = pd.read_csv(file)
    df = df.drop.na()
    # Set target variable
    y = df['GEDI_AGB']
    # Drop columns not required
    df = df.drop(columns = ['GEDI_X', 'GEDI_Y', 'Source'])
    # Isolate predictor variables
    pred_vars = df.drop(columns = ['GEDI_AGB'])
    results = {}
    # Perform linear regression
    for var in pred_vars.columns:
        X_column = pred_vars[[var]]
        model = LinearRegression()
        model.fit(X_column, y)
        y_pred = model.predict(X_column)
        # Calculate regression statistics
        r2 = r2_score(y, y_pred)
        results[var] = {
            'coef': model.coef_[0],
            'intercept': model.intercept_,
            'r2_score': r2}
    print(results)




# Code #############################################################################################################
if __name__ == '__main__':

    # Determine command line arguments
    parser = argparse.ArgumentParser(description = 'Assess data stored in csv for filtered training data')
    parser.add_argument('--count', action = 'store_true', help = 'Boolean to count footprints/intersects in file')
    parser.add_argument('--file', help = 'Filename of csv with directory, to return n footprints by unique value (site_year)')
    parser.add_argument('--regress', action = 'store_true', help = 'Instruction to perform linear regression with predictor variables')
    args = parser.parse_args()
    
    if args.count:
        #file = '/home/s1949330/data/diss_data/pred_vars/input_final/All_MODEL_INPUT_GEO_FINAL.csv'
        sample_year(args.file)

    if args.regress:
        var_regress(args.file)
        