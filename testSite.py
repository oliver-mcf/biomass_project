
# Train Model with Site Input Data

# Libraries ########################################################################################################
from libraries import *


# Objects & Methods ################################################################################################
from trainModel import isolate_data, train_model, test_model, model_scatter, model_hist, variable_importance


# Code #############################################################################################################
if __name__ == '__main__':

    # Define the new command line arguments
    parser = argparse.ArgumentParser(description = 'Train and test a Random Forest model using data from different sites')
    parser.add_argument('--label', required = True, help = 'Predictor label (e.g., Landsat, Sentinel, Palsar, All, Test)')
    parser.add_argument('--folder', required = True, help = 'Folder to save results')
    parser.add_argument('--trainSite', required = True, help = 'Site for training data')
    parser.add_argument('--testSite', required = True, help = 'Site for testing data')
    parser.add_argument('--trees', type = int, default = 200, help = 'Number of trees in the random forest')
    parser.add_argument('--split', type = float, default = 0.3, help = 'Split ratio for input site data')
    args = parser.parse_args()

    # Isolate target and predictor variables for training data
    train_filename = f'/home/s1949330/Documents/scratch/diss_data/pred_vars/input_final/{args.trainSite}_MODEL_INPUT_FINAL.csv'
    y_train, x_train, coords_train = isolate_data(train_filename, args.label)

    # Isolate target and predictor variables for testing data
    test_filename = f'/home/s1949330/Documents/scratch/diss_data/pred_vars/input_final/{args.testSite}_MODEL_INPUT_FINAL.csv'
    y_test, x_test, coords_test = isolate_data(test_filename, args.label)

    # Train the model
    train_model(x_train, y_train, args.label, args.trees, args.folder)

    # Test the model
    y_pred, stats_dict = test_model(x_test, y_test, args.folder, args.label)

    # Visualize model performance
    model_scatter(y_test, y_pred, args.folder, args.label, model = 0, single_output = True)
    model_hist(y_test, y_pred, args.folder, args.label, model = 0, single_output = True)

    # Store variable importances
    variable_importance(args.folder, args.label, x_train.columns)

    # Print statistics
    print(f'R: {stats_dict["R"]}')
    print(f'R2: {stats_dict["R2"]:.3f}')
    print(f'Bias: {stats_dict["Bias"]:.3f} Mg/ha')
    print(f'MAE: {stats_dict["MAE"]:.3f} Mg/ha   /   {(stats_dict["MAE"] / np.mean(y_test)) * 100:.3f} %')
    print(f'RMSE: {stats_dict["RMSE"]:.3f} Mg/ha   /   {(stats_dict["RMSE"] / np.mean(y_test)) * 100:.3f} %')
    print(f'LM/F/P: {stats_dict["LM"]:.0f} / {stats_dict["F"]:.0f} / {stats_dict["P"]:.2f}')