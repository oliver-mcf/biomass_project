
# Train Model with Input Data

# Libraries ########################################################################################################
from libraries import *


# Objects & Methods ################################################################################################
def isolate(filename, label):
    df = pd.read_csv(filename)
    df = df.drop(columns = ['Source'])
    df = df.dropna()
    y = df['GEDI_AGB']
    coords = df[['GEDI_X', 'GEDI_Y']]
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

def split(x, y, split_ratio):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = split_ratio)
    return x_train, x_test, y_train, y_test

def train(x_train, y_train, label, trees, folder):
    print('Running Random Forest Algorithm ...')
    rf = RandomForestRegressor(n_estimators = trees, random_state = random.seed())
    rf.fit(x_train, y_train)
    model_filename = f'/home/s1949330/Documents/scratch/diss_data/model/{folder}/{label}_RF_MODEL.joblib'
    joblib.dump(rf, model_filename)

def test(x_test, y_test, folder, label):
    model_filename = f'/home/s1949330/Documents/scratch/diss_data/model/{folder}/{label}_RF_MODEL.joblib'
    rf = joblib.load(model_filename)
    y_pred = rf.predict(x_test)
    residuals = y_pred - y_test
    constant = sm.add_constant(y_test)
    white_test = sm.stats.diagnostic.het_white(residuals, constant)
    print(f'R: {round(stats.pearsonr(y_test, y_pred)[0], 3)}')
    print(f'R2: {r2_score(y_test, y_pred):.3f}')
    print(f'Bias: {np.sum(y_pred - y_test) / y_pred.shape[0]:.3f} Mg/ha')
    print(f'MAE: {np.mean(np.abs(y_test - y_pred)):.3f} Mg/ha   |   {(np.mean(np.abs(y_test - y_pred)) / np.mean(y_test)) * 100:.3f} %')
    print(f'RMSE: {sqrt(mean_squared_error(y_test, y_pred)):.3f} Mg/ha  |   {(sqrt(mean_squared_error(y_test, y_pred)) / np.mean(y_test)) * 100:.3f} %')
    print(f'LM/F/P: {white_test[0]:.0f} / {white_test[2]:.0f} / {white_test[1]:.2f}')
    return y_pred

def variable_importance(folder, label, var_names):
    model_filename = f'/home/s1949330/Documents/scratch/diss_data/model/{folder}/{label}_RF_MODEL.joblib'
    rf = joblib.load(model_filename)
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    output_filename = f'/home/s1949330/Documents/scratch/diss_data/model/{folder}/{label}_VARIABLE_IMPORTANCE.csv'
    with open(output_filename, mode = 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(['Variable', 'Importance (%)'])
        for idx in sorted_idx:
            scaled_importance = importances[idx] * 100
            writer.writerow([var_names[idx], scaled_importance])

def scatter(y_test, y_pred, folder, label):
    mask = (y_test < 300) & (y_pred < 300)
    y_test = y_test[mask]
    y_pred = y_pred[mask]
    # Plot scatter
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize = (14, 12))
    ax.scatter(y_test, y_pred, color = 'lightsteelblue', alpha = 0.4, edgecolor = 'steelblue')
    ax.plot([0,300], [0,300], ls = 'solid', color = 'k')
    # PLot line of best fit
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
    plt.savefig(f'/home/s1949330/Documents/scratch/diss_data/model/{folder}/{label}_MODEL_TEST.png')
    plt.show()

def hist(y_test, y_pred, folder, label):
    mask = (y_test <= 300) & (y_pred <= 300)
    y_test = y_test[mask]
    y_pred = y_pred[mask]
    #Plot 2D histogram
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize = (10, 8))
    plt.hist2d(y_test, y_pred, bins = (100,100), cmap = 'viridis', cmin = 1)
    cbar = plt.colorbar(shrink = 0.75)
    ax.plot([0,300], [0,300], ls = 'solid', color = 'k')
    # PLot line of best fit
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
    plt.savefig(f'/home/s1949330/Documents/scratch/diss_data/model/{folder}/{label}_MODEL_TEST.png')
    plt.show()


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
    args = parser.parse_args()
    
    # Isolate target and predictor variables
    if args.site == '':
        input_filename = '/home/s1949330/Documents/scratch/diss_data/pred_vars/input_final/MODEL_INPUT_FILTER.csv'
    else:
        input_filename = f'/home/s1949330/Documents/scratch/diss_data/pred_vars/input_final/{args.site}_MODEL_INPUT_FILTER.csv'
    y, x, coords = isolate(input_filename, args.label)

    # Split data for model training
    if args.sample:
        sample_indices = random.sample(range(len(x)), k = int(0.1 * len(x)))
        x_sampled = x.iloc[sample_indices]
        y_sampled = y.iloc[sample_indices]
        print('Random Training Data Sample Size: {:,}'.format(len(x_sampled)))
        x_train, x_test, y_train, y_test = split(x_sampled, y_sampled, split_ratio = args.split)
    else: 
        x_train, x_test, y_train, y_test = split(x, y, split_ratio = args.split)

    # Save model training data
    training_data = pd.concat([coords, pd.DataFrame(x_train), pd.DataFrame({'y_train': y_train})], axis = 1).dropna(subset = ['y_train'])
    testing_data = pd.concat([coords, pd.DataFrame(x_test), pd.DataFrame({'y_test': y_test})], axis = 1).dropna(subset = ['y_test'])
    if args.site == '':
        training_data.to_csv(f'/home/s1949330/Documents/scratch/diss_data/model/{args.folder}/{args.label}_MODEL_TRAINING.csv', index = False)
        testing_data.to_csv(f'/home/s1949330/Documents/scratch/diss_data/model/{args.folder}/{args.label}_MODEL_TESTING.csv', index = False)
    else:
        training_data.to_csv(f'/home/s1949330/Documents/scratch/diss_data/model/{args.folder}/{args.site}_{args.label}_MODEL_TRAINING.csv', index = False)
        testing_data.to_csv(f'/home/s1949330/Documents/scratch/diss_data/model/{args.folder}/{args.site}_{args.label}_MODEL_TESTING.csv', index = False)

    # Train RF model
    train(x_train, y_train, args.label, args.trees, args.folder)

    # Test RF model performance
    y_pred = test(x_test, y_test, args.folder, args.label)
    scatter(y_test, y_pred, args.folder, args.label)
    hist(y_test, y_pred, args.folder, args.label)

    # VStore variable importance
    variable_importance(args.folderargs.label, x.columns)


