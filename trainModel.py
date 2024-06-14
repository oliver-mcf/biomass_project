
# Train Model with Input Data

# Libraries ########################################################################################################
from libraries import *


# Objects & Methods ################################################################################################
def isolate(filename):
    df = pd.read_csv(filename)
    df = df.drop(columns = ['Source'])
    df = df.drop(columns = ['01_NDVI_Median', '02_NDVI_Median', '03_NDVI_Median', '04_NDVI_Median'])
    #df = df.loc[:, ~ df.columns.str.contains('_NDVI_Median')]
    #na_counts = df.isnull().sum()
    #max_na_column = na_counts.idxmax()
    #max_na_count = na_counts.max()
    #print(max_na_column, max_na_count)
    df = df.dropna()
    input = df['GEDI_AGB']
    coords = df[['GEDI_X', 'GEDI_Y']]
    predictors = df.drop(columns = ['GEDI_X', 'GEDI_Y', 'GEDI_AGB'])
    print('Available Points for Model Training: {:,}'.format(len(predictors)))
    return input, predictors, coords

def split(predictors, input,  split):             
    predictors_train, predictors_test, input_train, input_test = train_test_split(predictors, input, test_size = split)
    return(predictors_train, predictors_test, input_train, input_test)

def train(predictors_train, input_train):
    print('Running Random Forest Algorithm ...')
    rf = RandomForestRegressor(n_estimators = 200, random_state = random.seed())
    rf.fit(predictors_train, input_train)
    joblib.dump(rf, '/home/s1949330/Documents/scratch/diss_data/model/RF_MODEL_2.joblib')
    print('Successful Model Training')
    return

def test(predictors_test, input_test):
    rf = joblib.load('/home/s1949330/Documents/scratch/diss_data/model/RF_MODEL_2.joblib')
    output = rf.predict(predictors_test)
    print('Successful Model Testing')
    r2 = stats.pearsonr(input_test, output)
    mae = np.mean(input_test - output)
    rmse = sqrt(mean_squared_error(input_test, output))
    mae_percentage = (mae / np.mean(input_test)) * 100
    rmse_percentage = (rmse / np.mean(input_test)) * 100
    print(f'R2 = {r2[0]:.3f}')
    print(f'MAE = {mae:.3f} (Mg/ha) \t MAE% = {mae_percentage:.2f}')
    print(f'RMSE = {rmse:.3f} (Mg/ha) \t RMSE% = {rmse_percentage:.2f}')
    return(output)

def white(output, input_test):
    residuals = output - input_test
    constant = sm.add_constant(input_test)
    white_test = sm.stats.diagnostic.het_white(residuals, constant)
    print(f'Lagrange Multiplier: {white_test[0]:.2f}')
    print(f'f-value: {white_test[2]:.2f}')
    print(f'p-value: {white_test[1]:.2f}')

def variable_importance(var_names):
    rf = joblib.load('/home/s1949330/Documents/scratch/diss_data/model/RF_MODEL_2.joblib')
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    sorted_importances = importances[sorted_idx]
    print("\n Variable Importances:")
    for i, importance in enumerate(sorted_importances):
        scaled_importance = importance * 100
        print("{}: {} - {:.2f}%".format(i + 1, var_names[sorted_idx[i]], scaled_importance))    

def convert_bytes(number):
    '''Function to convert file size to known units'''
    for x in ['bytes', 'KB', 'MB', 'GB']:
        if number < 1024.0:
            return "%3.1f %s" % (number, x)
        number /= 1024.0
    return

def file_size(file):
    '''Function to calculate filesize'''
    if os.path.isfile(file):
        file_info = os.stat(file)
        return convert_bytes(file_info.st_size)
    return


# Code #############################################################################################################
if __name__ == '__main__':

    # Start CPU runtime
    start = time.process_time()

    # Isolate model input and predictor variables
    input, predictors, coords = isolate('/home/s1949330/Documents/scratch/diss_data/model/MODEL_INPUT.csv')

    # Split data for model training
    predictors_train, predictors_test, input_train, input_test = split(predictors, input, split = 0.3)

    # Save model training data to visualise
    training_data = pd.concat([coords, pd.DataFrame({'input_train': input_train})], axis = 1).dropna(subset = ['input_train'])
    testing_data = pd.concat([coords, pd.DataFrame({'input_test': input_test})], axis = 1).dropna(subset = ['input_test'])
    training_data.to_csv('/home/s1949330/Documents/scratch/diss_data/model/LANDSAT_MODEL_TRAINING.csv', index = False)
    testing_data.to_csv('/home/s1949330/Documents/scratch/diss_data/model/LANDSAT_MODEL_TESTING.csv', index = False)

    # Train RF model
    train(predictors_train, input_train)

    # Test RF model performance
    output = test(predictors_test, input_test)
    white(output, input_test)

    # Highlight RF model variable importances
    var_names = predictors.columns.tolist()
    variable_importance(var_names)

    # Visualise RF model test
    plt.rcParams['font.family'] = 'Arial'
    max_value = max(max(input_test), max(output))
    plt.figure(figsize = (8, 6))
    plt.plot([0, max_value], [0, max_value], ls = '-', color = 'k')
    plt.xlim([0, max_value])
    plt.ylim([0, max_value])
    plt.hist2d(input_test, output, bins = (40, 40), cmap = 'cividis', cmin = 1)
    cbar = plt.colorbar(shrink = 0.75)
    plt.title('Model Test')
    plt.xlabel('Observed AGB (Mg/ha)')
    plt.ylabel('Predicted AGB (Mg/ha)')
    plt.savefig('/home/s1949330/Documents/scratch/diss_data/model/MODEL_TEST.png')
    plt.show()

    # Calculate CPU runtime and RAM usage
    print(f"CPU runtime: {round(((time.process_time() - start) / 60), 2)} minutes")
    ram = psutil.Process().memory_info().rss
    print(f"RAM usage: {convert_bytes(ram)}")




