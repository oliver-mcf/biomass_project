
# Train Model with Available Input Data

# Libraries ########################################################################################################
from libraries import *


# Objects & Methods ################################################################################################
def isolate_data(filename, label, filter = False):
    '''Isolate Predictor Variables for Various Model Configurations'''
    # Read input file
    df = pd.read_csv(filename)
    if filter == True:
        source = df['Source']
    df = df.drop(columns = ['Source'])
    df = df.dropna()
    df = df[(df != 0).all(axis = 1)]
    # Isolate target variables
    y = df['GEDI_AGB']
    coords = df[['GEDI_X', 'GEDI_Y']]
    # Set conditions for predictor variable inclusion
    if label == "Landsat":
        landsat_cols = [col for col in df.columns if col.startswith(('SRTM', 'SR', 'YR', 'T', 'NDVI'))]
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
    if filter == True:
        df_label = pd.concat([source, y, x, coords], axis = 1)
        return df_label
    else:
        print('Training Data Sample Size: {:,}'.format(len(x)))
        return y, x, coords

def save_splits(x_train, y_train, x_test, y_test, coords, args, fold = None):
    '''Save Training and Testing Subsets'''
    # Isolate subsets
    training_data = pd.concat([coords, pd.DataFrame(x_train), pd.DataFrame({'y_train': y_train})], axis = 1).dropna(subset = ['y_train'])
    testing_data = pd.concat([coords, pd.DataFrame(x_test), pd.DataFrame({'y_test': y_test})], axis = 1).dropna(subset = ['y_test'])
    # Consider site-specific condition
    training_data_filename = f'/home/s1949330/scratch/diss_data/model/{args.folder}/{args.label}_MODEL_TRAINING.csv'
    testing_data_filename = f'/home/s1949330/scratch/diss_data/model/{args.folder}/{args.label}_MODEL_TESTING.csv'
    # Consider iterative model training and testing
    if fold is not None:
        training_data_filename = training_data_filename.replace('.csv', f'_FOLD{fold + 1}.csv')
        testing_data_filename = testing_data_filename.replace('.csv', f'_FOLD{fold + 1}.csv')
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
    model_filename = f'/home/s1949330/scratch/diss_data/model/{folder}/{label}_RF_MODEL.joblib'
    if fold is not None:
        model_filename = model_filename.replace('.joblib', f'_FOLD{fold + 1}.joblib')
    joblib.dump(rf, model_filename)

def test_model(x_test, y_test, folder, label, fold = None):
    '''Test Model with Withheld Subset of Available Training Data'''
    # Read model
    model_filename = f'/home/s1949330/scratch/diss_data/model/{folder}/{label}_RF_MODEL.joblib'
    if fold is not None:
        model_filename = model_filename.replace('.joblib', f'_FOLD{fold + 1}.joblib')
    print(f"Loading: {os.path.basename(model_filename)}")
    rf = joblib.load(model_filename)
    # Statistically analyse model output
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
    format_stats = {key: round(value, 3) for key, value in stats_dict.items()}
    pprint(format_stats)
    return y_pred, stats_dict

def variable_importance(folder, label, var_names, fold = None):
    '''Save Predictor Variable Importance in Model'''
    # Read model
    model_filename = f'/home/s1949330/scratch/diss_data/model/{folder}/{label}_RF_MODEL.joblib'
    if fold is not None:
        model_filename = model_filename.replace('.joblib', f'_FOLD{fold + 1}.joblib')
    rf = joblib.load(model_filename)
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    # Write variable importances to csv
    output_filename = f'/home/s1949330/scratch/diss_data/model/{folder}/{label}_VARIABLE_IMPORTANCE.csv'
    if fold is not None:
        output_filename = output_filename.replace('.csv', f'_FOLD{fold + 1}.csv')
    with open(output_filename, mode = 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(['Variable', 'Importance (%)'])
        for idx in sorted_idx:
            scaled_importance = importances[idx] * 100
            writer.writerow([var_names[idx], scaled_importance])

def model_scatter(y_test, y_pred, folder, label, model, geo):
    '''Plot Scatter of Observed and Predicted Values'''
    # Constrain values to model sensitivity
    mask = (y_test < 120) & (y_pred < 120)
    y_test = y_test[mask]
    y_pred = y_pred[mask]
    # Plot scatter
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize = (8,8))
    ax.scatter(y_test, y_pred, marker = '.', color = 'lightsteelblue')
    if geo == 'PALSAR':
        upper = 100
        step = 5
    elif geo == 'COVER':
        upper = 150
        step = 10
    else:
        upper = 300
        step = 10
    ax.plot([0, upper], [0, upper], ls = 'solid', color = 'k')
    # Plot line of best fit
    slope, intercept = np.polyfit(y_test, y_pred, 1)
    best_fit = slope * np.array(y_test) + intercept
    ax.plot(y_test, best_fit, ls = 'solid', color = 'red', label = 'Linear')
    ax.set_xlim([0, upper])
    ax.set_ylim([0, upper])
    ax.set_xticks(np.arange(0, upper + step, step))
    ax.set_yticks(np.arange(0, upper + step, step))
    ax.set_xlabel('Observed AGB (Mg/ha)')
    ax.set_ylabel('Predicted AGB (Mg/ha)')
    ax.set_title(f'{label} Model - Observed vs Predicted')
    fig_name = f'/home/s1949330/scratch/diss_data/model/{folder}/{label}_MODEL_SCATTER_FOLD{model + 1}.png'
    plt.savefig(fig_name, dpi = 300)
    plt.close(fig)

def model_hist(y_test, y_pred, folder, label, model, geo):
    '''Plot Histogram of Observed and Predicted Values'''
    # Constrain values to model sensitivity
    mask = (y_test < 120) & (y_pred < 120)
    y_test = y_test[mask]
    y_pred = y_pred[mask]
    # Plot histogram
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize = (10,8))
    if geo == 'PALSAR':
        bins = 25
        upper = 150
        step = 5
    elif geo == 'COVER':
        bins = 25
        upper = 150
        step = 10
    else:
        bins = 50
        upper = 300
        step = 10    
    hist = plt.hist2d(y_test, y_pred, bins = (bins,bins), cmap = 'BuPu', cmin = 1)
    #plt.colorbar(shrink = 0.75)
    cbar = plt.colorbar(hist[3], shrink = 0.5, ticks = np.arange(0, upper, 2))
    ax.plot([0,upper], [0,upper], ls = 'solid', color = 'k')
    # Plot line of best fit
    slope, intercept = np.polyfit(y_test, y_pred, 1)
    best_fit = slope * np.array(y_test) + intercept
    ax.plot(y_test, best_fit, ls = 'solid', color = 'red', label = 'Linear')
    ax.set_xlim([0, upper])
    ax.set_ylim([0, upper])
    ax.set_xticks(np.arange(0, upper + step, step))
    ax.set_yticks(np.arange(0, upper + step, step))
    ax.set_xlabel('Observed AGB (Mg/ha)')
    ax.set_ylabel('Predicted AGB (Mg/ha)')
    ax.set_title(f'{label} Model - Observed vs Predicted')
    fig_name = f'/home/s1949330/scratch/diss_data/model/{folder}/{label}_MODEL_HIST_FOLD{model + 1}.png'
    plt.savefig(fig_name, dpi = 300)
    plt.close(fig)

def cross_validation(x, y, sample, kfolds, label, trees, folder, geo):
    '''Train Model with K-Fold Cross-Validation'''
    # Configure k-fold cross validation
    kf = KFold(n_splits = kfolds, shuffle = True, random_state = random.seed())
    stats_list = []
    if sample:
        sample_indices = random.sample(range(len(x)), k = int(0.05 * len(x)))
        x = x.iloc[sample_indices]
        y = y.iloc[sample_indices]
        print('Training Data Sample Size: {:,}'.format(len(x)))
    # Perform cross validation
    for fold, (train_index, test_index) in enumerate(kf.split(x)):
        print(f'Fold {fold + 1}/{kfolds}')
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # Train model
        train_model(x_train, y_train, label, trees, folder, fold)
        # Test model
        y_pred, stats_dict = test_model(x_test, y_test, folder, label, fold)
        stats_dict['Fold'] = fold + 1
        stats_list.append(stats_dict)
        # Save splits
        save_splits(x_train, y_train, x_test, y_test, x.index.to_series(), args, fold)
        # Visualise model performance
        model_scatter(y_test, y_pred, folder, label, model = fold, geo = geo)
        model_hist(y_test, y_pred, folder, label, model = fold, geo = geo)
        # Store variable importances
        variable_importance(folder, label, x.columns, fold)
        fold += 1
    # Save statistics to csv
    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv(f'/home/s1949330/scratch/diss_data/model/{folder}/{label}_KFOLD_STATS.csv', index = False)


# Code #############################################################################################################
if __name__ == '__main__':
    
    # Define command line arguments
    parser = argparse.ArgumentParser(description = 'Random Forest Model Training with K-Fold Cross Validation')
    parser.add_argument('--label', required = True, help = 'Predictor label (e.g., Landsat, Sentinel, Palsar, All)')
    parser.add_argument('--site', type = str, help = 'Condition to train and test model for specific site')
    parser.add_argument('--folder', required = True, help = 'Folder to save results')
    parser.add_argument('--kfolds', type = int, default = 10, help = 'Number of k-folds for cross-validation')
    parser.add_argument('--sample', action = 'store_true', help = 'Adopt a smaller sample size of the available training data')
    parser.add_argument('--trees', type = int, default = 200, help = 'Number of trees in the random forest')
    parser.add_argument('--split', type = float, default = 0.3, help = 'Test split ratio')
    parser.add_argument('--geo', help = 'Geolocation filtering condition: PALSAR or COVER or blank')
    args = parser.parse_args()

    # Isolate target and filtered predictor variables
    input_filename = (f'/home/s1949330/scratch/diss_data/pred_vars/input_final/All_{args.site}_EXTRACT_FINAL_{args.geo}.csv'
                      if args.site else 
                      f'/home/s1949330/scratch/diss_data/pred_vars/input_final/All_EXTRACT_FINAL_{args.geo}.csv')
    y, x, coords = isolate_data(input_filename, args.label)

    # Perform k-fold cross validation for model training
    cross_validation(x, y, args.sample, args.kfolds, args.label, args.trees, args.folder, args.geo)


