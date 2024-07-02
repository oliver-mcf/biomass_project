
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
        sample_indices = random.sample(range(len(x)), k = int(0.10 * len(x)))
        x_sampled = x.iloc[sample_indices]
        y_sampled = y.iloc[sample_indices]
        print('Training Data Sample Size: {:,}'.format(len(x_sampled)))
        x_train, x_test, y_train, y_test = train_test_split(x_sampled, y_sampled, test_size = split_ratio)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = split_ratio)
    return x_train, x_test, y_train, y_test

def save_splits(x_train, y_train, x_test, y_test, coords, args, fold = None):
    '''Save Training and Testing Subsets'''
    # Isolate subsets
    training_data = pd.concat([coords, pd.DataFrame(x_train), pd.DataFrame({'y_train': y_train})], axis = 1).dropna(subset = ['y_train'])
    testing_data = pd.concat([coords, pd.DataFrame(x_test), pd.DataFrame({'y_test': y_test})], axis = 1).dropna(subset = ['y_test'])
    
    # Consider site-specific condition
    if args.site == '':
        training_data_filename = f'/home/s1949330/Documents/scratch/diss_data/model/{args.folder}/{args.label}_MODEL_TRAINING.csv'
        testing_data_filename = f'/home/s1949330/Documents/scratch/diss_data/model/{args.folder}/{args.label}_MODEL_TESTING.csv'
    else:
        training_data_filename = f'/home/s1949330/Documents/scratch/diss_data/model/{args.folder}/{args.site}_{args.label}_MODEL_TRAINING.csv'
        testing_data_filename = f'/home/s1949330/Documents/scratch/diss_data/model/{args.folder}/{args.site}_{args.label}_MODEL_TESTING.csv'
    
    # If fold is provided, append fold number to the filenames
    if fold is not None:
        training_data_filename = training_data_filename.replace('.csv', f'_FOLD{fold}.csv')
        testing_data_filename = testing_data_filename.replace('.csv', f'_FOLD{fold}.csv')
    
    # Write splits to csv
    training_data.to_csv(training_data_filename, index = False)
    testing_data.to_csv(testing_data_filename, index = False)

def train_model(x_train, y_train, label, trees, folder, fold = None):
    '''Train Model with Subset of Available Training Data'''
    print('Running Random Forest Algorithm ...')
    # Set random forest parameters
    rf = RandomForestRegressor(n_estimators = trees, random_state = random.seed())
    rf.fit(x_train, y_train)
    # Save trained model
    model_filename = f'/home/s1949330/Documents/scratch/diss_data/model/{folder}/{label}_RF_MODEL.joblib'
    if fold is not None:
        model_filename = model_filename.replace('.joblib', f'_FOLD{fold}.joblib')
    joblib.dump(rf, model_filename)

def test_model(x_test, y_test, folder, label, fold = None):
    '''Test Model with Withheld Subset of Available Training Data'''
    # Read model
    model_filename = f'/home/s1949330/Documents/scratch/diss_data/model/{folder}/{label}_RF_MODEL.joblib'
    if fold is not None:
        model_filename = model_filename.replace('.joblib', f'_FOLD{fold}.joblib')
    rf = joblib.load(model_filename)
    # Statistically analyse model output
    y_pred = rf.predict(x_test)
    residuals = y_pred - y_test
    constant = sm.add_constant(y_test)
    white_test = sm.stats.diagnostic.het_white(residuals, constant)
    stats_dict = {
        'R': round(stats.pearsonr(y_test, y_pred)[0], 3),
        'R2': r2_score(y_test, y_pred),
        'Bias': np.sum(y_pred - y_test) / y_pred.shape[0],
        'MAE': np.mean(np.abs(y_test - y_pred)),
        'RMSE': sqrt(mean_squared_error(y_test, y_pred)),
        'LM': white_test[0],
        'F': white_test[2],
        'P': white_test[1]
    }
    print(f'R: {stats_dict["R"]}')
    print(f'R2: {stats_dict["R2"]:.3f}')
    print(f'Bias: {stats_dict["Bias"]:.3f} Mg/ha')
    print(f'MAE: {stats_dict["MAE"]:.3f} Mg/ha   /   {(stats_dict["MAE"] / np.mean(y_test)) * 100:.3f} %')
    print(f'RMSE: {stats_dict["RMSE"]:.3f} Mg/ha   /   {(stats_dict["RMSE"] / np.mean(y_test)) * 100:.3f} %')
    print(f'LM/F/P: {stats_dict["LM"]:.0f} / {stats_dict["F"]:.0f} / {stats_dict["P"]:.2f}')
    return y_pred, stats_dict

def variable_importance(folder, label, var_names, fold = None):
    '''Save Predictor Variable Importance in Model'''
    # Read model
    model_filename = f'/home/s1949330/Documents/scratch/diss_data/model/{folder}/{label}_RF_MODEL.joblib'
    if fold is not None:
        model_filename = model_filename.replace('.joblib', f'_FOLD{fold}.joblib')
    rf = joblib.load(model_filename)
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    # Write variable importances to csv
    output_filename = f'/home/s1949330/Documents/scratch/diss_data/model/{folder}/{label}_VARIABLE_IMPORTANCE.csv'
    if fold is not None:
        output_filename = output_filename.replace('.csv', f'_FOLD{fold}.csv')
    with open(output_filename, mode = 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(['Variable', 'Importance (%)'])
        for idx in sorted_idx:
            scaled_importance = importances[idx] * 100
            writer.writerow([var_names[idx], scaled_importance])

def model_scatter(y_test, y_pred, folder, label, fold = None):
    '''Plot Observed and Predicted Values'''
    # Constrain values to model sensitivity
    mask = (y_test < 300) & (y_pred < 300)
    y_test = y_test[mask]
    y_pred = y_pred[mask]
    # Plot scatter
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize = (14, 12))
    ax.scatter(y_test, y_pred, color = 'lightsteelblue', alpha = 0.4, edgecolor = 'steelblue')
    ax.plot([0, 300], [0, 300], ls = 'solid', color = 'k')
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
    ax.set_title(f'{label} Model - Observed vs Predicted')
    if fold is not None:
        plt.savefig(f'/home/s1949330/Documents/scratch/diss_data/model/{folder}/{label}_MODEL_SCATTER_FOLD{fold}.png')
    else:
        plt.savefig(f'/home/s1949330/Documents/scratch/diss_data/model/{folder}/{label}_MODEL_SCATTER.png')
    plt.close(fig)

def model_hist(y_test, y_pred, folder, label, fold = None):
    '''Plot Histogram of Prediction Residuals'''
    residuals = y_test - y_pred
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize = (14, 12))
    ax.hist(residuals, bins = 50, color = 'lightsteelblue', edgecolor = 'steelblue')
    ax.set_xlabel('Prediction Residuals (Mg/ha)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{label} Model - Residuals Histogram')
    if fold is not None:
        plt.savefig(f'/home/s1949330/Documents/scratch/diss_data/model/{folder}/{label}_MODEL_HIST_FOLD{fold}.png')
    else:
        plt.savefig(f'/home/s1949330/Documents/scratch/diss_data/model/{folder}/{label}_MODEL_HIST.png')
    plt.close(fig)

def cross_validation(x, y, k, label, trees, folder):
    '''Perform K-Fold Cross-Validation and Save Statistics'''
    # Configure k-fold cross validation
    kf = KFold(n_splits = k, shuffle = True, random_state = random.seed())
    stats_list = []
    # Perform cross validation
    for fold, (train_index, test_index) in enumerate(kf.split(x)):
        print(f'Fold {fold + 1}/{k}')
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # Train model
        train_model(x_train, y_train, label, trees, folder, fold + 1)
        # Test model
        y_pred, stats_dict = test_model(x_test, y_test, folder, label, fold + 1)
        stats_dict['fold'] = fold + 1
        stats_list.append(stats_dict)
        # Save splits
        save_splits(x_train, y_train, x_test, y_test, x.index.to_series(), args, fold + 1)
    # Save statistics to csv
    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv(f'/home/s1949330/Documents/scratch/diss_data/model/{folder}/{label}_KFOLD_STATS.csv', index = False)


# Code #############################################################################################################
if __name__ == '__main__':
    
    # Define command line arguments
    parser = argparse.ArgumentParser(description = 'Random Forest Model Training and Evaluation')
    parser.add_argument('--label', required = True, help = 'Predictor label (e.g., Landsat, Sentinel, Palsar, All, Test)')
    parser.add_argument('--site', default = '', help = 'Site-specific identifier')
    parser.add_argument('--folder', required = True, help = 'Folder to save results')
    parser.add_argument('--sample', action = 'store_true', help = 'Adopt a smaller sample size of the available training data')
    parser.add_argument('--trees', type = int, default = 200, help = 'Number of trees in the random forest, default 200')
    parser.add_argument('--split', type = float, default = 0.3, help = 'Test split ratio, default 0.3')
    parser.add_argument('--kfolds', type = int, default = 5, help = 'Number of k-folds for cross-validation, default, 5')
    parser.add_argument('--fold_model', type = int, default = None, help = 'Fold number to evaluate specific model')
    args = parser.parse_args()

    # Isolate target and predictor variables
    input_filename = f'/home/s1949330/Documents/scratch/diss_data/pred_vars/input_final/{args.site}_MODEL_INPUT_FINAL.csv' if args.site else '/home/s1949330/Documents/scratch/diss_data/pred_vars/input_final/MODEL_INPUT_FINAL.csv'
    y, x, coords = isolate_data(input_filename, args.label)

    if args.kfolds:
        # Perform k-fold cross validation 
        cross_validation(x, y, args.kfolds, args.label, args.trees, args.folder)
    else:
        if args.fold_model is not None:
            # Evaluate a specific fold model
            fold = args.fold_model
            model_filename = f'/home/s1949330/Documents/scratch/diss_data/model/{args.folder}/{args.label}_RF_MODEL_FOLD{fold}.joblib'
            y_test = pd.read_csv(f'/home/s1949330/Documents/scratch/diss_data/model/{args.folder}/{args.label}_MODEL_TESTING_FOLD{fold}.csv')['y_test']
            x_test = pd.read_csv(f'/home/s1949330/Documents/scratch/diss_data/model/{args.folder}/{args.label}_MODEL_TESTING_FOLD{fold}.csv').drop(columns = ['y_test'])
            y_pred, _ = test_model(x_test, y_test, args.folder, args.label, fold)
            model_scatter(y_test, y_pred, args.folder, args.label, fold)
            model_hist(y_test, y_pred, args.folder, args.label, fold)
            variable_importance(args.folder, args.label, x.columns, fold)
        else:
            # Split data for model training
            x_train, x_test, y_train, y_test = split_data(x, y, args.split, args.sample)
            save_splits(x_train, y_train, x_test, y_test, coords, args)
            # Train RF model
            train_model(x_train, y_train, args.label, args.trees, args.folder)
            # Test RF model performance
            y_pred, stats_dict = test_model(x_test, y_test, args.folder, args.label)
            model_scatter(y_test, y_pred, args.folder, args.label)
            model_hist(y_test, y_pred, args.folder, args.label)
            # Save variable importances
            variable_importance(args.folder, args.label, x.columns)
