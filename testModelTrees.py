
# Test Random Forest Model Parameters

# Libraries ########################################################################################################
from libraries import *


# Objects & Methods ################################################################################################
from trainModel import isolate_data

def split_data(x, y, split_ratio, sample = False):
    '''Subset Available Data for Training and Testing'''
    # Sample available data and split for model training and testing
    if sample:
        sample_indices = random.sample(range(len(x)), k = int(0.10 * len(x)))
        x_sampled = x.iloc[sample_indices]
        y_sampled = y.iloc[sample_indices]
        print('Training Data Sample Size: {:,}'.format(len(x_sampled)))
        x_train, x_test, y_train, y_test = train_test_split(x_sampled, y_sampled, test_size = split_ratio)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = split_ratio)
    return x_train, x_test, y_train, y_test

def param_train(x_train, y_train, trees):
    rf = RandomForestRegressor(n_estimators = trees, random_state = random.seed())
    rf.fit(x_train, y_train)
    return rf

def param_test(rf, x_test, y_test):
    y_pred = rf.predict(x_test)
    residuals = y_pred - y_test
    constant = sm.add_constant(y_test)
    white_test = sm.stats.diagnostic.het_white(residuals, constant)
    stats_dict = {
        'R2 (r2_score)': r2_score(y_test, y_pred),
        'R2 (rf.score)': rf.score(x_test, y_test),
        'R2 (variance)': explained_variance_score(y_test, y_pred),
        'Bias': np.sum(y_pred - y_test) / y_pred.shape[0],
        'MAE': np.mean(np.abs(y_test - y_pred)),
        'MAE%': np.mean(np.abs(y_test - y_pred)) / np.mean(y_test) * 100,
        'RMSE': sqrt(mean_squared_error(y_test, y_pred)),
        'RMSE%': sqrt(mean_squared_error(y_test, y_pred)) / np.mean(y_test) * 100,
        'R': stats.pearsonr(y_test, y_pred)[0],
        'LM': white_test[0],
        'F': white_test[2],
        'P': white_test[1]}
    return stats_dict, y_pred


# Code #############################################################################################################
if __name__ == '__main__':
    
    # Define command line arguments
    parser = argparse.ArgumentParser(description = 'Assess Random Forest Model Performance by Number of Trees')
    parser.add_argument('--label', type = str, choices = ['Landsat', 'Sentinel', 'Palsar', 'All'], required = True, help = 'Keyword for selecting predictor variables')
    parser.add_argument('--test', action = 'store_true', help = 'Adopt a smaller sample size of the available training data for testing')
    parser.add_argument('--subset', type = float, default = 0.50, help = 'Proportion of original dataset kept for testing, 0-1')
    parser.add_argument('--split', type = float, default = 0.3, help = 'Ratio for splitting the data into training and testing sets')
    args = parser.parse_args()

    # Isolate target and predictor variables
    input_filename = '/home/s1949330/Documents/scratch/diss_data/pred_vars/input_final/All_EXTRACT_FINAL_COVER.csv'
    y, x, _ = isolate_data(input_filename, args.label)

    # Generate random subset of training data
    if args.test:
        sample_indices = random.sample(range(len(x)), k = int(args.subset * len(x)))
        x = x.iloc[sample_indices]
        y = y.iloc[sample_indices]
        print('Training Data Sample Size: {:,}'.format(len(x)))

    # Split sampled data for model training
    x_train, x_test, y_train, y_test = split_data(x, y, split_ratio = args.split)

    # List of n_estimators to iterate over
    n_estimators_list = [100, 200, 300, 400, 500]

    # Iterate over n_estimators, train and test the model
    for n_estimators in n_estimators_list:
        rf = param_train(x_train, y_train, n_estimators)
        stats_dict, y_pred = param_test(rf, x_test, y_test)
        print('Trees:', {n_estimators})
        pprint(stats_dict)
