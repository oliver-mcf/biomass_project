
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
        x = df.loc[:, 'SR_B2_Median' : 'SRTM_mTPI']
    elif label == "Sentinel":
        x = df.loc[:, 'SRTM_mTPI' : 'VH_05']
    elif label == "Palsar":
        x = df.loc[:, 'SR_B2_Median' : 'HHHV_Index']
    elif label == "All":
        x = df.loc[:, 'SR_B2_Median' : 'HHHV_Index']
    else:
        raise ValueError("Invalid keyword specified.")
    print('Available Points for Model Training: {:,}'.format(len(x)))
    return y, x, coords

def split(x, y, split_ratio):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = split_ratio)
    return x_train, x_test, y_train, y_test

def train(x_train, y_train, label, site = ''):
    print('Running Random Forest Algorithm ...')
    rf = RandomForestRegressor(n_estimators = 200, random_state = random.seed())
    rf.fit(x_train, y_train)
    model_filename = f'/home/s1949330/Documents/scratch/diss_data/model/{site}_{label}_RF_MODEL.joblib' if site else f'/home/s1949330/Documents/scratch/diss_data/model/{label}_RF_MODEL.joblib'
    joblib.dump(rf, model_filename)

def test(x_test, y_test, label, site = ''):
    model_filename = f'/home/s1949330/Documents/scratch/diss_data/model/{site}_{label}_RF_MODEL.joblib' if site else f'/home/s1949330/Documents/scratch/diss_data/model/{label}_RF_MODEL.joblib'
    rf = joblib.load(model_filename)
    y_pred = rf.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    print(f'R2: {r2:.3f}')
    print(f'Bias: {np.sum(y_pred - y_test) / y_pred.shape[0]:.3f} Mg/ha')
    print(f'MAE: {np.mean(np.abs(y_test - y_pred)):.3f} Mg/ha')
    print(f'MAE%: {(np.mean(np.abs(y_test - y_pred)) / np.mean(y_test)) * 100:.3f} %')
    print(f'RMSE: {sqrt(mean_squared_error(y_test, y_pred)):.3f} Mg/ha')
    print(f'RMSE%: {(sqrt(mean_squared_error(y_test, y_pred)) / np.mean(y_test)) * 100:.3f} %')
    residuals = y_pred - y_test
    constant = sm.add_constant(y_test)
    white_test = sm.stats.diagnostic.het_white(residuals, constant)
    print(f'LM/F/P: {white_test[0]:.0f} / {white_test[2]:.0f} / {white_test[1]:.2f}')
    return y_pred

def variable_importance(label, var_names, site = ''):
    model_filename = f'/home/s1949330/Documents/scratch/diss_data/model/{site}_{label}_RF_MODEL.joblib' if site else f'/home/s1949330/Documents/scratch/diss_data/model/{label}_RF_MODEL.joblib'
    rf = joblib.load(model_filename)
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    output_filename = f'/home/s1949330/Documents/scratch/diss_data/model/{site}_{label}_VARIABLE_IMPORTANCE.csv' if site else f'/home/s1949330/Documents/scratch/diss_data/model/{label}_VARIABLE_IMPORTANCE.csv'
    with open(output_filename, mode = 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(['Variable', 'Importance (%)'])
        for idx in sorted_idx:
            scaled_importance = importances[idx] * 100
            writer.writerow([var_names[idx], scaled_importance])

'''''''''
def model_plot(y_test, y_pred):
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize = (10, 8))
    hist = ax.hist2d(y_test, y_pred, bins = 30, cmap = 'cividis', cmin = 1)
    ax.plot([0, 200], [0, 200], ls = 'solid', color = 'k')
    sat = y_pred < 200
    slope, intercept = np.polyfit(y_test[sat], y_pred[sat], 1)
    best_fit = slope * np.array(y_test[sat]) + intercept
    ax.plot(y_test[sat], best_fit, ls = 'solid', color = 'red')
    ax.set_xlim([0, 200])
    ax.set_ylim([0, 200])
    ax.set_xticks(np.arange(0, 220, 20))
    ax.set_yticks(np.arange(0, 220, 20))
    cbar = plt.colorbar(hist[3], ax = ax, label = 'Count', shrink = 0.6)
    ax.set_xlabel('Observed AGB (Mg/ha)')
    ax.set_ylabel('Predicted AGB (Mg/ha)')
    plt.savefig('MODEL_TEST.png')
    plt.show()
'''''''''


# Code #############################################################################################################
if __name__ == '__main__':
    
    # Define command line arguments
    parser = argparse.ArgumentParser(description = 'Random Forest Model Training and Evaluation')
    parser.add_argument('label', type = str, choices = ['Landsat', 'Sentinel', 'Palsar', 'All'], help = 'Keyword for selecting predictor variables')
    parser.add_argument('--split', type = float, default = 0.3, help = 'Ratio for splitting the data into training and testing sets (Default: 0.3)')
    parser.add_argument('--site', type = str, default = '', help = 'Site name to filter CSV files (optional)')
    args = parser.parse_args()
    
    # Isolate target and predictor variables
    input_filename = f'/home/s1949330/Documents/scratch/diss_data/model/csv/{args.site}_MODEL_INPUT.csv' if args.site else '/home/s1949330/Documents/scratch/diss_data/model/csv/MODEL_INPUT.csv'
    y, x, coords = isolate(input_filename, args.label)

    # Split data for model training
    x_train, x_test, y_train, y_test = split(x, y, split_ratio = args.split)

    # Save model training data
    training_data = pd.concat([coords, pd.DataFrame(x_train), pd.DataFrame({'y_train': y_train})], axis = 1).dropna(subset = ['y_train'])
    testing_data = pd.concat([coords, pd.DataFrame(x_test), pd.DataFrame({'y_test': y_test})], axis = 1).dropna(subset = ['y_test'])
    if args.site:
        training_data_filename = f'/home/s1949330/Documents/scratch/diss_data/model/{args.site}_{args.label}_MODEL_TRAINING.csv'
        testing_data_filename = f'/home/s1949330/Documents/scratch/diss_data/model/{args.site}_{args.label}_MODEL_TESTING.csv'
    else:
        training_data_filename = f'/home/s1949330/Documents/scratch/diss_data/model/{args.label}_MODEL_TRAINING.csv'
        testing_data_filename = f'/home/s1949330/Documents/scratch/diss_data/model/{args.label}_MODEL_TESTING.csv'
    training_data.to_csv(training_data_filename, index = False)
    testing_data.to_csv(testing_data_filename, index = False)

    # Train RF model
    train(x_train, y_train, args.label, args.site)

    # Test RF model performance
    y_pred = test(x_test, y_test, args.label, args.site)

    # Variable importance
    variable_importance(args.label, x.columns, args.site)


