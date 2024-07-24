
# Train Model with Site Input Data

# Libraries ########################################################################################################
from libraries import *


# Objects & Methods ################################################################################################
from trainModel import *

def spatial_cv(train_filename, test_filename, label, trees, folder, geo):
    '''Train Model with Spatial-Validation'''
    # Load data from separate sites
    y_train, x_train, _ = isolate_data(train_filename, label)
    y_test, x_test, _ = isolate_data(test_filename, label)
    stats_list = []
    # Train model
    train_model(x_train, y_train, label, trees, folder)
    # Test model
    y_pred, stats_dict = test_model(x_test, y_test, folder, label)
    stats_dict['Fold'] = 1
    stats_list.append(stats_dict)
    # Visualise model performance
    model_scatter(y_test, y_pred, folder, label, model = 0, geo = geo)
    model_hist(y_test, y_pred, folder, label, model = 0, geo = geo)
    # Store variable importances
    variable_importance(folder, label, x_train.columns)
    # Save statistics to csv
    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv(f'/home/s1949330/Documents/scratch/diss_data/model/{folder}/{label}_STATS.csv', index = False)



# Code #############################################################################################################
if __name__ == '__main__':

    # Define the new command line arguments
    parser = argparse.ArgumentParser(description = 'Train and test a Random Forest model using data from different sites')
    parser.add_argument('--label', default = 'All', help = 'Predictor label (e.g., Landsat, Sentinel, Palsar, All)')
    parser.add_argument('--folder', required = True, help = 'Folder to save results')
    parser.add_argument('--trainSite', required = True, help = 'Site for training data')
    parser.add_argument('--testSite', required = True, help = 'Site for testing data')
    parser.add_argument('--trees', type = int, default = 200, help = 'Number of trees in the random forest')
    parser.add_argument('--split', type = float, default = 0.3, help = 'Split ratio for input site data')
    parser.add_argument('--geo', help = 'Limit the output figure params to suit either geolocation: PALSAR or COVER')
    args = parser.parse_args()

    # Define train and test site filenames
    if args.geo:
        train_site = f'/home/s1949330/Documents/scratch/diss_data/pred_vars/input_final/All_{args.trainSite}_EXTRACT_FINAL_{args.geo}.csv'
        test_site = f'/home/s1949330/Documents/scratch/diss_data/pred_vars/input_final/All_{args.testSite}_EXTRACT_FINAL_{args.geo}.csv'
    else:
        train_site = f'/home/s1949330/Documents/scratch/diss_data/pred_vars/input_final/All_{args.trainSite}_EXTRACT_FINAL.csv'
        test_site = f'/home/s1949330/Documents/scratch/diss_data/pred_vars/input_final/All_{args.testSite}_EXTRACT_FINAL.csv'

    # Perform spatial cross validation
    spatial_cv(train_site, test_site, args.label, args.trees, args.folder, args.geo)

