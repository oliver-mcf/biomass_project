
# Predict Biomass with Model

# Libraries ########################################################################################################
from libraries import *


# Objects & Methods ################################################################################################
from extractData import GeoTiff

def filter_vars(vars, year):
    '''Filter Variables to Match Model Predictors'''
    # Store variable names to retain
    ref_csv = f'/home/s1949330/Documents/scratch/diss_data/pred_vars/input_final/All_FILTERED_VARIABLES.csv'
    df = pd.read_csv(ref_csv)
    df['Retained'] = f'{year}_' + df['Retained'].astype(str)
    var_names = df['Retained'].tolist()
    print('Predictor Variables to Match: ', len(var_names))
    # Retain variables found in reference df  
    year_vars = []
    for var in tqdm(vars, desc = "FILTER"):
        base = os.path.splitext(os.path.basename(var))[0]
        if base in var_names:
            year_vars.append(var)
    return year_vars

def prepare_vars1(pred_vars):
    '''Prepare Data for Model Predictions'''
    # Iterate through predictor variables to read data
    #flat_dataset = []
    #for var in tqdm(pred_vars, desc = "PREPARE"):
    for var in pred_vars:
        pred = GeoTiff(var)
        data = pred.data
        print(pred.nX, pred.nY)
        data_flat = data.flatten()
        data_flat[np.isnan(data_flat)] = -999
        #flat_dataset.append(data_flat)
    # Align flattened predictor variables
    #pred_flat = np.stack(flat_dataset, axis = -1)
    #print(pred_flat.shape)
    #return pred_flat

def prepare_vars(pred_vars, ref_var, site):
    '''Prepare Data for Model Predictions'''
    # Set common dimensions for variables
    if site == 'MGR':
        common_nX, common_nY = ref_var.nX, ref_var.nY
    elif site == 'TKW':
        common_nX, common_nY = ref_var.nX, ref_var.nY - 1
    print(f'Common Variable Dimensions: {common_nX} x {common_nY}') 
    # Iterate through predictor variables to read data
    flat_dataset = []
    #for var in tqdm(pred_vars, desc = "PREPARE"):
    for var in pred_vars:
        pred = GeoTiff(var)
        data = pred.data[:common_nY, :common_nX]
        data_flat = data.flatten()
        data_flat[np.isnan(data_flat)] = -999
        flat_dataset.append(data_flat)
    # Align flattened predictor variables
    pred_flat = np.stack(flat_dataset, axis = -1)
    print(pred_flat.shape)
    return pred_flat

def predict_agb(pred_flat, batch, folder, model):
    # Configure batch process
    rf = joblib.load(f'/home/s1949330/Documents/scratch/diss_data/model/{folder}/All_RF_MODEL_FOLD{model}.joblib')
    pred_agb = np.empty((pred_flat.shape[0],), dtype = float)
    batch_size = int(batch * pred_flat.shape[0])
    num_batches = int(math.ceil(pred_flat.shape[0] / batch_size))
    # Predict in batches
    for i in tqdm(range(num_batches), desc = "PREDICT"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, pred_flat.shape[0])
        pred_agb[start_idx:end_idx] = rf.predict(pred_flat[start_idx:end_idx])
    return pred_agb

def pred_hist(pred_data, bins, site, year, folder):
    '''Plot Histogram from Array'''
    # Mask data for sensitivity
    pred_data[pred_data > 200] = 200
    # Plot the histogram
    plt.rcParams['font.family'] = 'Arial'
    plt.figure(figsize = (12, 10))
    plt.hist(pred_data, bins = bins, color = 'teal', edgecolor = 'teal', alpha = 0.6)
    plt.title(f'Histogram of Predicted Aboveground Biomass for {site}, 20{year}')
    plt.ylabel('Frequency')
    plt.xlabel('Biomass (Mg/ha)')
    plt.grid(True)
    plt.savefig(f'/home/s1949330/Documents/scratch/diss_data/model/{folder}/predict/{site}_{year}_PREDICT_AGB_HIST.png', dpi = 300)
    plt.close()

def pred_map(pred_data, ref_var, folder, site, year):
    '''Reshape Array of Predictions to Produce Map'''
    # Mask data for sensitivity
    pred_data[pred_data > 200] = 200
    # Visualise predicted biomass map
    plt.rcParams['font.family'] = 'Arial'
    agb_map = np.reshape(pred_data, (ref_var.nY, ref_var.nX))
    plt.figure(figsize = (24, 20))
    plt.imshow(agb_map, cmap = 'Greens', vmin = 0, vmax = 200)    
    plt.title(f'Predicted Agboveground Biomass for {site}, 20{year}')
    cbar = plt.colorbar(shrink = 0.75)
    cbar.set_label('Biomass (Mg/ha)')
    cbar.set_ticks(np.arange(0, 210, 10))
    plt.savefig(f'/home/s1949330/Documents/scratch/diss_data/model/{folder}/predict/{site}_{year}_PREDICT_AGB.png', dpi = 300)
    plt.close()


# Code #############################################################################################################
if __name__ == '__main__':

    # Define command line arguments
    parser = argparse.ArgumentParser(description = "Extract data for a given site over given year(s).")
    parser.add_argument("--model", type = int, required = True, choices = [1, 2, 3, 4, 5], help = "Name of trained model to use for predictions")
    parser.add_argument("--folder", type = str, required = True, help = "Directory containing model.")
    parser.add_argument("--site", type = str, required = True, help = "Study site by SEOSAW abbreviation.")
    #parser.add_argument("--year", type = int, required = True, help = "End of austral year, eg: for Aug 2019 to July 2020, give 20 for 2020.")
    parser.add_argument("--batch", type = float, default = 0.05, help = 'Proportion of study site to compute predictions between 0-1 g: 0.05 for 5 percent')
    parser.add_argument("--bins", type = int, default = 20, help = 'Number of bins to arrange data in histogram')
    args = parser.parse_args()

    # Iterate over years
    year_list = ['20', '21', '22', '23']
    for year in year_list:
        print(f'Currently Predicting Biomass for 20{year}...')

        # Isolate filtered predictor variables
        year_vars = glob(f'/home/s1949330/Documents/scratch/diss_data/pred_vars/{args.site}/{year}_*.tif')
        vars = filter_vars(year_vars, year)
        srtm_vars = glob(f'/home/s1949330/Documents/scratch/diss_data/pred_vars/{args.site}/SRTM_*.tif')
        pred_vars = sorted(vars + srtm_vars)
        
        # Identify one reference variable
        one_var = [f for f in pred_vars if os.path.basename(f) == f"{year}_HHHV_Ratio.tif"]
        ref_var = GeoTiff(one_var[0])

        # Prepare predictor variables
        pred_flat = prepare_vars(pred_vars, ref_var, args.site)
        print(pred_flat.shape)

        # Batch process model predictions
        pred_agb = predict_agb(pred_flat, args.batch, args.folder, args.model)

        # Sanity check biomass predictions 
        print('Max:', np.max(pred_agb))
        print('Mean:', np.mean(pred_agb))
        print('Median:', np.median(pred_agb))
        print('Min:', np.min(pred_agb))

        # Plot histogram of biomass predictions
        pred_hist(pred_agb, args.bins, args.site, year, args.folder)

        # Visualise biomass prediction
        pred_map(pred_agb, ref_var, args.folder, args.site, year)

        # Save biomass map as geotiff
