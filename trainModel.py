
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

def train(x_train, y_train, label):
    print('Running Random Forest Algorithm ...')
    rf = RandomForestRegressor(n_estimators = 100, random_state = random.seed())
    rf.fit(x_train, y_train)
    model_filename = f'/home/s1949330/Documents/scratch/diss_data/model/{label}_RF_MODEL.joblib'
    joblib.dump(rf, model_filename)

def test(x_test, y_test, label):
    model_filename = f'/home/s1949330/Documents/scratch/diss_data/model/{label}_RF_MODEL.joblib'
    rf = joblib.load(model_filename)
    y_pred = rf.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    print(f'R2: {r2:.3f}')
    print(f'Bias: {np.sum(y_pred - y_test) / y_pred.shape[0]:.3f} Mg/ha')
    print(f'MAE: {np.mean(np.abs(y_test - y_pred)):.3f} Mg/ha')
    print(f'MAE%: {(np.mean(np.abs(y_test - y_pred)) / np.mean(y_test)) * 100:.3f} %')
    print(f'RMSE: {sqrt(mean_squared_error(y_test, y_pred)):.3f} Mg/ha')
    print(f'RMSE%: {(sqrt(mean_squared_error(y_test, y_pred)) / np.mean(y_test)) * 100:.3f} %')
    #residuals = y_pred - y_test
    #constant = sm.add_constant(y_test)
    #white_test = sm.stats.diagnostic.het_white(residuals, constant)
    #print(f'LM/F/P: {white_test[0]:.0f} / {white_test[2]:.0f} / {white_test[1]:.2f}')
    return y_pred

def variable_importance(label, var_names):
    model_filename = f'/home/s1949330/Documents/scratch/diss_data/model/{label}_RF_MODEL.joblib'
    rf = joblib.load(model_filename)
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    output_filename = f'/home/s1949330/Documents/scratch/diss_data/model/{label}_VARIABLE_IMPORTANCE.csv'
    with open(output_filename, mode = 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(['Variable', 'Importance (%)'])
        for idx in sorted_idx:
            scaled_importance = importances[idx] * 100
            writer.writerow([var_names[idx], scaled_importance])

def model_plot(y_test, y_pred, label):
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(y_test, y_pred, color = 'blue', alpha = 0.5, edgecolor = 'k')
    ax.plot([0, 200], [0, 200], ls = 'solid', color = 'k')
    slope, intercept = np.polyfit(y_test, y_pred, 1)
    best_fit = slope * np.array(y_test) + intercept
    ax.plot(y_test, best_fit, ls = 'solid', color = 'red', label = 'Line of Best Fit')
    ax.set_xlim([0, 200])
    ax.set_ylim([0, 200])
    ax.set_xticks(np.arange(0, 210, 10))
    ax.set_yticks(np.arange(0, 210, 10))
    ax.set_xlabel('Observed AGB (Mg/ha)')
    ax.set_ylabel('Predicted AGB (Mg/ha)')
    ax.legend()
    plt.savefig(f'/home/s1949330/Documents/scratch/diss_data/model/{label}_MODEL_TEST.png')
    plt.show()



# Code #############################################################################################################
if __name__ == '__main__':
    
    # Define command line arguments
    parser = argparse.ArgumentParser(description = 'Random Forest Model Training and Evaluation')
    parser.add_argument('label', type = str, choices = ['Landsat', 'Sentinel', 'Palsar', 'All'], help = 'Keyword for selecting predictor variables')
    parser.add_argument('--split', type = float, default = 0.3, help = 'Ratio for splitting the data into training and testing sets; default = 0.3')
    parser.add_argument('--trees', type = str, default = 200, help = 'Number of estimators to train random forest model; default = 200')
    args = parser.parse_args()
    
    # Isolate target and predictor variables
    input_filename = '/home/s1949330/Documents/scratch/diss_data/model/MODEL_INPUT_FILTER.csv'
    y, x, coords = isolate(input_filename, args.label)

    # Split data for model training
    x_train, x_test, y_train, y_test = split(x, y, split_ratio = args.split)

    # Save model training data
    training_data = pd.concat([coords, pd.DataFrame(x_train), pd.DataFrame({'y_train': y_train})], axis = 1).dropna(subset = ['y_train'])
    testing_data = pd.concat([coords, pd.DataFrame(x_test), pd.DataFrame({'y_test': y_test})], axis = 1).dropna(subset = ['y_test'])
    training_data_filename = f'/home/s1949330/Documents/scratch/diss_data/model/{args.label}_MODEL_TRAINING.csv'
    testing_data_filename = f'/home/s1949330/Documents/scratch/diss_data/model/{args.label}_MODEL_TESTING.csv'
    training_data.to_csv(training_data_filename, index = False)
    testing_data.to_csv(testing_data_filename, index = False)

    # Train RF model
    train(x_train, y_train, args.label)

    # Test RF model performance
    y_pred = test(x_test, y_test, args.label)
    model_plot(y_test, y_pred, args.label)

    # Variable importance
    variable_importance(args.label, x.columns)


