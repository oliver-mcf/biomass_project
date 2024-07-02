
# Train Model with Input Data

# Libraries ########################################################################################################
from libraries import *


# Objects & Methods ################################################################################################
def isolate_data(filename, label):
    '''Isolate Predictor Variables for Various Model Configurations'''
    # Read input file
    df = pd.read_csv(filename)
    df = df.drop(columns = ['Source'])
    df = df.dropna()
    # Isolate target variables
    y = df['GEDI_AGB']
    coords = df[['GEDI_X', 'GEDI_Y']]
    # Set conditions for predictor variable inclusion
    if label == "Landsat":
        landsat_cols = [col for col in df.columns if col.startswith(('SRTM', 'SR', 'T'))]
        x = df.loc[:, landsat_cols]
    elif label == "Sentinel":
        sentinel_cols = [col for col in df.columns if col.startswith(('SRTM', 'VV', 'VH'))]
        x = df.loc[:, sentinel_cols]
    elif label == "Palsar":
        palsar_cols = [col for col in df.columns if col.startswith(('SRTM', 'HH', 'HV'))]
        x = df.loc[:, palsar_cols]
    elif label == "All":
        exclude_substrings = ['GEDI_X', 'GEDI_Y', 'GEDI_AGB']
        combined_cols = [col for col in df.columns if not any(substring in col for substring in exclude_substrings)]
        x = df.loc[:, combined_cols]
    elif label == "Test":
        test_cols = [col for col in df.columns if col.startswith(('SRTM', 'T'))]
        x = df.loc[:, test_cols]
    print('Training Data Sample Size: {:,}'.format(len(x)))
    return y, x, coords

def split_data(x, y, split_ratio, sample = False):
    '''Subset Available Data for Training and Testing'''
    # Sample available data and split for model training and testing
    if sample:
        sample_indices = random.sample(range(len(x)), k = int(0.25 * len(x)))
        x_sampled = x.iloc[sample_indices]
        y_sampled = y.iloc[sample_indices]
        print('Training Data Sample Size: {:,}'.format(len(x_sampled)))
        x_train, x_test, y_train, y_test = train_test_split(x_sampled, y_sampled, split_ratio = split_ratio)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, split_ratio = split_ratio)
    return x_train, x_test, y_train, y_test

def save_splits(x_train, y_train, x_test, y_test, coords, args):
    '''Save Training and Testing Subsets'''
    # Isolate subsets
    training_data = pd.concat([coords, pd.DataFrame(x_train), pd.DataFrame({'y_train': y_train})], axis = 1).dropna(subset = ['y_train'])
    testing_data = pd.concat([coords, pd.DataFrame(x_test), pd.DataFrame({'y_test': y_test})], axis = 1).dropna(subset = ['y_test'])
    # Consider site specific condition
    if args.site == '':
        training_data_filename = f'/home/s1949330/Documents/scratch/diss_data/model/{args.folder}/{args.label}_MODEL_TRAINING.csv'
        testing_data_filename = f'/home/s1949330/Documents/scratch/diss_data/model/{args.folder}/{args.label}_MODEL_TESTING.csv'
    else:
        training_data_filename = f'/home/s1949330/Documents/scratch/diss_data/model/{args.folder}/{args.site}_{args.label}_MODEL_TRAINING.csv'
        testing_data_filename = f'/home/s1949330/Documents/scratch/diss_data/model/{args.folder}/{args.site}_{args.label}_MODEL_TESTING.csv'
    # Write splits to csv
    training_data.to_csv(training_data_filename, index = False)
    testing_data.to_csv(testing_data_filename, index = False)

def train_model(x_train, y_train, label, trees, folder):
    '''Train Model with Subset of Available Training Data'''
    print('Running Random Forest Algorithm ...')
    # Set random forest parameters
    rf = RandomForestRegressor(n_estimators = trees, random_state = random.seed())
    rf.fit(x_train, y_train)
    # Save trained model
    model_filename = f'/home/s1949330/Documents/scratch/diss_data/model/{folder}/{label}_RF_MODEL.joblib'
    joblib.dump(rf, model_filename)

def test_model(x_test, y_test, folder, label):
    '''Test Model with Witheld Subset of Available Training Data'''
    # Read model
    model_filename = f'/home/s1949330/Documents/scratch/diss_data/model/{folder}/{label}_RF_MODEL.joblib'
    rf = joblib.load(model_filename)
    # Statistically analyse model output
    y_pred = rf.predict(x_test)
    residuals = y_pred - y_test
    constant = sm.add_constant(y_test)
    white_test = sm.stats.diagnostic.het_white(residuals, constant)
    print(f'R: {round(stats.pearsonr(y_test, y_pred)[0], 3)}')
    print(f'R2: {r2_score(y_test, y_pred):.3f}')
    print(f'Bias: {np.sum(y_pred - y_test) / y_pred.shape[0]:.3f} Mg/ha')
    print(f'MAE: {np.mean(np.abs(y_test - y_pred)):.3f} Mg/ha   /   {(np.mean(np.abs(y_test - y_pred)) / np.mean(y_test)) * 100:.3f} %')
    print(f'RMSE: {sqrt(mean_squared_error(y_test, y_pred)):.3f} Mg/ha   /   {(sqrt(mean_squared_error(y_test, y_pred)) / np.mean(y_test)) * 100:.3f} %')
    print(f'LM/F/P: {white_test[0]:.0f} / {white_test[2]:.0f} / {white_test[1]:.2f}')
    return y_pred

def variable_importance(folder, label, var_names):
    '''Save Predictor Variable Importance in Model'''
    # Read model
    model_filename = f'/home/s1949330/Documents/scratch/diss_data/model/{folder}/{label}_RF_MODEL.joblib'
    rf = joblib.load(model_filename)
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    # Write variable importances to csv
    output_filename = f'/home/s1949330/Documents/scratch/diss_data/model/{folder}/{label}_VARIABLE_IMPORTANCE.csv'
    with open(output_filename, mode = 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(['Variable', 'Importance (%)'])
        for idx in sorted_idx:
            scaled_importance = importances[idx] * 100
            writer.writerow([var_names[idx], scaled_importance])

def model_scatter(y_test, y_pred, folder, label):
    '''Plot Observed and Predicted Values'''
    # Constrain values to model sensitivity
    mask = (y_test < 300) & (y_pred < 300)
    y_test = y_test[mask]
    y_pred = y_pred[mask]
    # Plot scatter
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize = (14, 12))
    ax.scatter(y_test, y_pred, color = 'lightsteelblue', alpha = 0.4, edgecolor = 'steelblue')
    ax.plot([0,300], [0,300], ls = 'solid', color = 'k')
    # Plot line of best fit
    slope, intercept = np.polyfit(y_test, y_pred, 1)
    best_fit = slope * np.array(y_test) + intercept
    ax.plot(y_test, best_fit, ls = 'solid', color = 'red', label = 'Linear')
    ax.set_xlim([0, 300])
    ax.set_ylim([0, 300])
    ax.set_xticks(np.arange(0, 320, 20))
    ax.set_yticks(np.arange(0, 320, 20))
    ax.set_xlabel('Observed AGB (Mg/ha)')
    ax.set_ylabel('Predicted AGB (Mg/ha)')
    ax.legend()
    plt.savefig(f'/home/s1949330/Documents/scratch/diss_data/model/{folder}/{label}_MODEL_SCATTER.png')
    #plt.show()

def model_hist(y_test, y_pred, folder, label):
    '''Plot density of Observed and Predicted Values'''
    # Constrain values to model sensitivity
    mask = (y_test <= 300) & (y_pred <= 300)
    y_test = y_test[mask]
    y_pred = y_pred[mask]
    # Plot 2D histogram
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize = (10, 8))
    plt.hist2d(y_test, y_pred, bins = (100,100), cmap = 'turbo', cmin = 10)
    cbar = plt.colorbar(shrink = 0.75)
    ax.plot([0,300], [0,300], ls = 'solid', color = 'k')
    # Plot line of best fit
    slope, intercept = np.polyfit(y_test, y_pred, 1)
    best_fit = slope * np.array(y_test) + intercept
    ax.plot(y_test, best_fit, ls = 'solid', color = 'red', label = 'Linear')
    ax.set_xlim([0, 200])
    ax.set_ylim([0, 200])
    ax.set_xticks(np.arange(0, 220, 20))
    ax.set_yticks(np.arange(0, 220, 20))
    ax.set_xlabel('Observed AGB (Mg/ha)')
    ax.set_ylabel('Predicted AGB (Mg/ha)')
    ax.legend()
    plt.savefig(f'/home/s1949330/Documents/scratch/diss_data/model/{folder}/{label}_MODEL_HIST.png')
    #plt.show()

def cross_validation(x, y, coords, args):
    '''Perform a K-Fold Cross Validation for RF Model'''
    # Prepare for k-fold outputs
    kf = KFold(n_splits = args.kfolds, shuffle = True, random_state = random.seed())
    fold = 1
    all_y_test = []
    all_y_pred = []
    validation_stats = []
    # Perform k-fold segmentations of available training data
    for train_index, test_index in kf.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # Save model training data for each fold
        training_data = pd.concat([coords.iloc[train_index], pd.DataFrame(x_train), pd.DataFrame({'y_train': y_train})], axis = 1).dropna(subset = ['y_train'])
        testing_data = pd.concat([coords.iloc[test_index], pd.DataFrame(x_test), pd.DataFrame({'y_test': y_test})], axis = 1).dropna(subset = ['y_test'])
        training_data.to_csv(f'/home/s1949330/Documents/scratch/diss_data/model/{args.folder}/{args.label}_MODEL_TRAINING_FOLD{fold}.csv', index = False)
        testing_data.to_csv(f'/home/s1949330/Documents/scratch/diss_data/model/{args.folder}/{args.label}_MODEL_TESTING_FOLD{fold}.csv', index = False)
        # Train RF model for each fold
        rf = RandomForestRegressor(n_estimators = args.trees, random_state = random.seed())
        rf.fit(x_train, y_train)
        joblib.dump(rf, f'/home/s1949330/Documents/scratch/diss_data/model/{args.folder}/{args.label}_RF_MODEL_FOLD{fold}.joblib')
        # Test RF model performance for each fold
        y_pred = rf.predict(x_test)
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)
        fold += 1
    # Calculate performance metrics across all folds
    all_y_test = np.array(all_y_test)
    all_y_pred = np.array(all_y_pred)
    residuals = all_y_pred - all_y_test
    constant = sm.add_constant(all_y_test)
    white_test = sm.stats.diagnostic.het_white(residuals, constant)
    validation_stats.append({'R': round(stats.pearsonr(all_y_test, all_y_pred)[0], 3),
                             'R2': round(r2_score(all_y_test, all_y_pred), 3),
                             'Bias': round(np.sum(all_y_pred - all_y_test) / all_y_pred.shape[0], 3), 
                             'MAE': round(mean_absolute_error(all_y_test, all_y_pred), 3),
                             'MAE%': round((sqrt(mean_squared_error(y_test, y_pred)) / np.mean(y_test)) * 100),
                             'RMSE': round(sqrt(mean_squared_error(all_y_test, all_y_pred)), 3),
                             'RMSE': round((sqrt(mean_squared_error(y_test, y_pred)) / np.mean(y_test)) * 100),
                             'LM/F/P': (white_test[0], white_test[2], white_test[1])})
    return all_y_test, all_y_pred, validation_stats


# Code #############################################################################################################
if __name__ == '__main__':
    
    # Define command line arguments
    parser = argparse.ArgumentParser(description = 'Random Forest Model Training and Evaluation')
    parser.add_argument('--label', type = str, choices = ['Landsat', 'Sentinel', 'Palsar', 'All'], help = 'Keyword for selecting predictor variables')
    parser.add_argument('--site', type = str, default = '', help = 'Study area for specific model training')
    parser.add_argument('--sample', action = 'store_true', help = 'Condition to use a sample of the training data')
    parser.add_argument('--split', type = float, default = 0.3, help = 'Ratio for splitting the data into training and testing sets; default = 0.3')
    parser.add_argument('--trees', type = int, default = 200, help = 'Number of estimators to train random forest model; default = 200')
    parser.add_argument('--folder', type = str, help = 'Directory folder for model outputs within: .../diss_data/model/')
    parser.add_argument('--kfolds', type = int, default = 5, help = 'Number of folds for K-Fold Cross-Validation; default = 5')
    args = parser.parse_args()
    
    # Isolate target and predictor variables
    if args.site == '':
        input_filename = '/home/s1949330/Documents/scratch/diss_data/pred_vars/input_final/MODEL_INPUT_FILTER.csv'
    else:
        input_filename = f'/home/s1949330/Documents/scratch/diss_data/pred_vars/input_final/{args.site}_MODEL_INPUT_FILTER.csv'
    y, x, coords = isolate_data(input_filename, args.label)

    # Split data for model training
    x_train, x_test, y_train, y_test = split_data(x, y, split_ratio = args.split, sample = args.sample)

    # Save model training data
    save_splits(x_train, y_train, x_test, y_test, coords, args)

    # Train RF model
    train_model(x_train, y_train, args.label, args.trees, args.folder)

    # Test RF model performance
    y_pred = test_model(x_test, y_test, args.folder, args.label)
    model_scatter(y_test, y_pred, args.folder, args.label)
    model_hist(y_test, y_pred, args.folder, args.label)

    # Store variable importance
    variable_importance(args.folder, args.label, x.columns)

    # Perform k-fold cross validation
    if args.kfolds > 1:
        all_y_test, all_y_pred, validation_stats = cross_validation(x, y, coords, args)
        print(f'K-Fold Cross-Validation Results: {validation_stats}')
