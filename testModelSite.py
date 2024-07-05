
# Train Model with Site Input Data

# Libraries ########################################################################################################
from libraries import *


# Objects & Methods ################################################################################################
from trainModel import *

def spatial_cv(train_filename, test_filename, kfolds, label, trees, folder, sample):
    '''Train Model with K-Fold Cross-Validation'''
    # Load data from separate sites
    y_train, x_train, _ = isolate_data(train_filename, label)
    y_test, x_test, _ = isolate_data(test_filename, label)
    # Configure k-fold cross validation
    kf = KFold(n_splits = kfolds, shuffle = True, random_state = random.seed())
    stats_list = []
    if sample:
        sample_indices = random.sample(range(len(x_train)), k = int(0.05 * len(x_train)))
        x_train = x_train.iloc[sample_indices]
        y_train = y_train.iloc[sample_indices]
        print('Training Data Sample Size: {:,}'.format(len(x_train)))
    # Perform cross validation
    for fold, (train_index, val_index) in enumerate(kf.split(x_train)):
        print(f'Fold {fold + 1}/{kfolds}')
        x_train_fold, x_val_fold = x_train.iloc[train_index], x_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        # Train model
        train_model(x_train_fold, y_train_fold, label, trees, folder, fold)
        # Test model
        y_pred, stats_dict = test_model(x_test, y_test, folder, label, fold)
        stats_dict['Fold'] = fold + 1
        stats_list.append(stats_dict)
        # Save splits
        save_splits(x_train_fold, y_train_fold, x_val_fold, y_val_fold, x_train.index.to_series(), args, fold)
        # Visualise model performance
        model_scatter(y_test, y_pred, folder, label, model = fold)
        model_hist(y_test, y_pred, folder, label, model = fold)
        # Store variable importances
        variable_importance(folder, label, x_train.columns, fold)
    # Save statistics to csv
    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv(f'/home/s1949330/Documents/scratch/diss_data/model/{folder}/{label}_KFOLD_STATS.csv', index = False)



# Code #############################################################################################################
if __name__ == '__main__':

    # Define the new command line arguments
    parser = argparse.ArgumentParser(description = 'Train and test a Random Forest model using data from different sites')
    parser.add_argument('--label', required = True, help = 'Predictor label (e.g., Landsat, Sentinel, Palsar, All)')
    parser.add_argument('--folder', required = True, help = 'Folder to save results')
    parser.add_argument('--trainSite', required = True, help = 'Site for training data')
    parser.add_argument('--testSite', required = True, help = 'Site for testing data')
    parser.add_argument('--kfolds', type = int, required = True, help = 'Number of k-folds for cross-validation')
    parser.add_argument('--sample', action = 'store_true', help = 'Adopt a smaller sample size of the available training data')
    parser.add_argument('--trees', type = int, default = 100, help = 'Number of trees in the random forest')
    parser.add_argument('--split', type = float, default = 0.3, help = 'Split ratio for input site data')
    args = parser.parse_args()

    # Define train and test site filenames
    train_site = f'/home/s1949330/Documents/scratch/diss_data/pred_vars/input_final/{args.label}_{args.trainSite}_MODEL_INPUT_FINAL.csv'
    test_site = f'/home/s1949330/Documents/scratch/diss_data/pred_vars/input_final/{args.label}_{args.testSite}_MODEL_INPUT_FINAL.csv'

    # Perform spatial k-fold cross validation
    spatial_cv(train_site, test_site, args.kfolds, args.label, args.trees, args.folder, args.sample)
 