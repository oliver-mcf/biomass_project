
# Train Model with Input Data

# Libraries ########################################################################################################
from libraries import *


# Objects & Methods ################################################################################################
def combine(csv_list):
    df_list = []
    for csv in csv_list:
        df = pd.read_csv(csv)
        df_list.append(df)
    combined_df = pd.concat(df_list, ignore_index = True)
    combined_df.to_csv('/home/s1949330/Documents/scratch/diss_data/model/MODEL_INPUT.csv', index = False)
    print('Successful merge of model input data')

def isolate(filename):
    df = pd.read_csv(filename)
    #df.dropna(inplace = True)
    input = df['GEDI_AGB']
    coords = df[['GEDI_X', 'GEDI_Y']]
    predictors = df.drop(columns = ['GEDI_X', 'GEDI_Y', 'GEDI_AGB'])
    print(len(predictors))
    return input, predictors, coords

def split(predictors, input,  split):             
    predictors_train, predictors_test, input_train, input_test = train_test_split(predictors, input, test_size = split)
    return(predictors_train, predictors_test, input_train, input_test)

def train(predictors_train, input_train):
    print('Running Random Forest Algorithm ...')
    rf = RandomForestRegressor(n_estimators = 500, random_state = random.seed())
    rf.fit(predictors_train, input_train)
    joblib.dump(rf, '/home/s1949330/Documents/scratch/diss_data/model/RF_MODEL.joblib')
    print('Successful Model Training')
    return

def test(predictors_test, input_test):
    rf = joblib.load('/home/s1949330/Documents/scratch/diss_data/model/RF_MODEL.joblib')
    output = rf.predict(predictors_test)
    print('Successful Model Testing')
    rValue = stats.pearsonr(input_test, output)
    pprint('MAE =', np.mean(input_test - output))
    pprint('R2 =', rValue[:1])
    pprint('RMSE =', sqrt(mean_squared_error(input_test, output)))
    return(output)

def white(output, input_test):
    residuals = output - input_test
    constant = sm.add_constant(input_test)
    white_test = sm.stats.diagnostic.het_white(residuals, constant)
    print(white_test)


# Code #############################################################################################################
if __name__ == '__main__':
    
    # Combine all model input data
    combine(glob(f'/home/s1949330/Documents/scratch/diss_data/*.csv'))

    # Isolate model input and predictor variables
    input, predictors, coords = isolate('/home/s1949330/Documents/scratch/diss_data/model/MODEL_INPUT.csv')

    # Split data for model training
    predictors_train, predictors_test, input_train, input_test = split(predictors, input, split = 0.3)

    # Save model training data to visualise
    training_data = pd.concat([coords, pd.DataFrame({'input_train': input_train})], axis = 1)
    testing_data = pd.concat([coords, pd.DataFrame({'input_test': input_test})], axis = 1)
    training_data.to_csv('/home/s1949330/Documents/scratch/diss_data/model/MODEL_TRAINING_DATA.csv', index = False)
    testing_data.to_csv('/home/s1949330/Documents/scratch/diss_data/model/MODEL_TESTING_DATA.csv', index = False)

    # Train RF model
    train(predictors_train, input_train)

    # Test RF model performance
    output = test(predictors_test, input_test)
    white(output, input_test)






