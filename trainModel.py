
# Train Model with Input Data

# Libraries ########################################################################################################
from libraries import *


# Objects & Methods ################################################################################################
def isolate(filename):
    df = pd.read_csv(filename)
    df = df.drop(columns = ['Source'])
    #df = df.drop(columns = ['01_NDVI_Median', '02_NDVI_Median', '03_NDVI_Median', '04_NDVI_Median'])       # FilteredMonths
    df = df.loc[:, ~ df.columns.str.contains('_NDVI_Median')]                                              # RemovedMonths
    df = df.dropna()
    input = df['GEDI_AGB']
    coords = df[['GEDI_X', 'GEDI_Y']]
    predictors = df.loc[:,'SR_B2_Median':'HV_05']
    # Landsat = SR_B2_Median : SRTM_mTPI
    # Sentinel = SR_B2_Median : VH_05
    # Palsar = SR_B2_Median : 'HV_05
    print('Available Points for Model Training: {:,}'.format(len(predictors)))
    return input, predictors, coords

def split(predictors, input,  split):             
    predictors_train, predictors_test, input_train, input_test = train_test_split(predictors, input, test_size = split)
    return(predictors_train, predictors_test, input_train, input_test)

def train(predictors_train, input_train):
    print('Running Random Forest Algorithm ...')
    rf = RandomForestRegressor(n_estimators = 200, random_state = random.seed())
    rf.fit(predictors_train, input_train)
    joblib.dump(rf, '/home/s1949330/Documents/scratch/diss_data/model/RF_MODEL.joblib')
    return

def test(predictors_test, input_test):
    rf = joblib.load('/home/s1949330/Documents/scratch/diss_data/model/RF_MODEL.joblib')
    output = rf.predict(predictors_test)
    r = stats.pearsonr(input_test, output)
    print(f'R: {r[0]:.3f}')
    print(f'Bias: {np.sum(output - input_test) / (output.shape[0]):.3f} Mg/ha')
    print(f'MAE: {np.mean(input_test - output):.3f} Mg/ha')
    print(f'MAE%: {(np.mean(input_test - output)) / (np.mean(input_test)) * 100:.3f} %')
    print(f'RMSE: {sqrt(mean_squared_error(input_test, output)):.3f} Mg/ha')
    print(f'RMSE%: {(sqrt(mean_squared_error(input_test, output))) / (np.mean(input_test)) * 100:.3f} %')
    residuals = output - input_test
    constant = sm.add_constant(input_test)
    white_test = sm.stats.diagnostic.het_white(residuals, constant)
    print(f'LM/F/P: {white_test[0]:.0f} / {white_test[2]:.0f} / {white_test[1]:.2f}')
    return(output)

def variable_importance(var_names):
    rf = joblib.load('/home/s1949330/Documents/scratch/diss_data/model/RF_MODEL.joblib')
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    with open('/home/s1949330/Documents/scratch/diss_data/model/VARIABLE_IMPORTANCE.csv', mode = 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(['Variable', 'Importance (%)'])
        for idx in sorted_idx:
            scaled_importance = importances[idx] * 100
            writer.writerow([var_names[idx], scaled_importance])   

def model_plot(input_test, output):
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize = (10, 8))
    hist = ax.hist2d(input_test, output, bins=30, cmap='cividis', cmin=1)
    ax.plot([0, 200], [0, 200], ls = 'solid', color = 'k')
    sat = output < 200
    slope, intercept = np.polyfit(input_test[sat], output[sat], 1)
    best_fit = slope * np.array(input_test[sat]) + intercept
    ax.plot(input_test[sat], best_fit, ls = 'solid', color = 'red')
    ax.set_xlim([0, 200])
    ax.set_ylim([0, 200])
    ax.set_xticks(np.arange(0, 220, 20))
    ax.set_yticks(np.arange(0, 220, 20))
    cbar = plt.colorbar(hist[3], ax=ax, label='Count', shrink=0.6)
    ax.set_xlabel('Observed AGB (Mg/ha)')
    ax.set_ylabel('Predicted AGB (Mg/ha)')
    plt.savefig('/home/s1949330/Documents/scratch/diss_data/model/MODEL_TEST.png')
    plt.show()


# Code #############################################################################################################
if __name__ == '__main__':

    # Isolate model input and predictor variables
    input, predictors, coords = isolate('/home/s1949330/Documents/scratch/diss_data/model/MODEL_INPUT.csv')

    # Split data for model training
    predictors_train, predictors_test, input_train, input_test = split(predictors, input, split = 0.25)

    # Save model training data to visualise
    training_data = pd.concat([coords, pd.DataFrame({'input_train': input_train})], axis = 1).dropna(subset = ['input_train'])
    testing_data = pd.concat([coords, pd.DataFrame({'input_test': input_test})], axis = 1).dropna(subset = ['input_test'])
    training_data.to_csv('/home/s1949330/Documents/scratch/diss_data/model/MODEL_TRAINING.csv', index = False)
    testing_data.to_csv('/home/s1949330/Documents/scratch/diss_data/model/MODEL_TESTING.csv', index = False)

    # Train RF model
    train(predictors_train, input_train)

    # Test RF model performance
    output = test(predictors_test, input_test)

    # Highlight RF model variable importances
    var_names = predictors.columns.tolist()
    variable_importance(var_names)

    # Visualise RF model test
    #model_plot(input_test, output)





