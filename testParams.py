
# Test Random Forest Model Parameters

# Libraries ########################################################################################################
from libraries import *
from trainModel import isolate, split


# Objects & Methods ################################################################################################
def param_train(x_train, y_train, trees):
    rf = RandomForestRegressor(n_estimators = trees, random_state = random.seed())
    rf.fit(x_train, y_train)
    return rf

def param_test(rf, x_test, y_test):
    y_pred = rf.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    return r2, y_pred


# Code #############################################################################################################
if __name__ == '__main__':
    
    # Define command line arguments
    parser = argparse.ArgumentParser(description = 'Random Forest Model Training and Evaluation')
    parser.add_argument('--label', type = str, choices = ['Landsat', 'Sentinel', 'Palsar', 'All', 'Test'], required = True, help = 'Keyword for selecting predictor variables')
    parser.add_argument('--split', type = float, default = 0.3, help = 'Ratio for splitting the data into training and testing sets; default = 0.3')
    parser.add_argument('--site', type = str, default = '', help = 'Site name to filter CSV files (optional)')
    args = parser.parse_args()

    # Isolate target and predictor variables
    if args.site == '':
        input_filename = '/home/s1949330/Documents/scratch/diss_data/pred_vars/input_final/MODEL_INPUT_FILTER.csv'
    else:
        input_filename = f'/home/s1949330/Documents/scratch/diss_data/pred_vars/input_merge/{args.site}_MODEL_INPUT_FILTER.csv'
    y, x, coords = isolate(input_filename, args.label)

    # Split sampled data for model training
    x_train, x_test, y_train, y_test = split(x, y, split_ratio = args.split)

    # List of n_estimators to iterate over
    n_estimators_list = [100, 200, 300]

    # Iterate over n_estimators, train and test the model
    for n_estimators in n_estimators_list:
        rf = param_train(x_train, y_train, n_estimators)
        r2, y_pred = param_test(rf, x_test, y_test)
        print(f'Trees: {n_estimators}, R2: {r2:.3f}')

