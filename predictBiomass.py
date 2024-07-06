
# Predict Biomass with Model

# Libraries ########################################################################################################
from libraries import *


# Objects & Methods ################################################################################################
from extractData import GeoTiff

def filter_vars(all_vars, label):
    '''Filter Variables to Match Model Predictors'''
    # Store variable names to retain
    ref_csv = f'/home/s1949330/Documents/scratch/diss_data/pred_vars/input_final/{label}_FILTERED_VARIABLES.csv'
    df = pd.read_csv(ref_csv)
    var_names = df['Retained']
    print('Predictor Variables to Match: ', len(var_names))
    # Retain variables found in reference df  
    pred_vars = []
    for var in tqdm(all_vars, desc = "FILTER"):
        base = os.path.splitext(os.path.basename(var))[0]
        if base in var_names:
            pred_vars.append(var)
    print('Predictor Variables Matched: ', len(pred_vars))
    return pred_vars

def prepare_vars(pred_vars):
    '''Prepare Data for Model Predictions'''
    # Iterate through predictor variables to read data
    flat_dataset = []
    for var in tqdm(pred_vars, desc = "PREPARE"):
        pred = GeoTiff(var)
        data = pred.data.flatten()
        data[np.isnan(data)] = -999
        flat_dataset.append(data)
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

def pred_hist(data, bins, site, year):
    '''Plot Histogram from Array'''
    # Define histogram properties
    counts, bin_edges = np.histogram(data, bins = bins)
    bin_middles = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    # Calculate smoothed line
    spline = make_interp_spline(bin_middles, counts, k = 3)
    bin_middles_smooth = np.linspace(bin_middles.min(), bin_middles.max(), 300)
    counts_smooth = spline(bin_middles_smooth)
    # Plot the histogram
    plt.figure(figsize = (12, 10))
    plt.hist(data, bins = bins, color = 'teal', edgecolor = 'teal', alpha = 0.6)
    plt.title(f'Histogram of Predicted Aboveground Biomass for {site}, 20{year}')
    plt.ylabel('Frequency')
    plt.xlabel('Biomass (Mg/ha)')
    plt.grid(True)
    # Plot smooth line
    plt.plot(bin_middles_smooth, counts_smooth, color = 'black', lw = 2)
    plt.show()

def pred_map(pred_agb, ref_var, folder, site, year):
    agb_map = np.reshape(pred_agb, (ref_var.nY, ref_var.nX))
    plt.figure(figsize = (24, 20))
    plt.imshow(agb_map, cmap = 'Greens')
    plt.title(f'Predicted Agboveground Biomass, 20{year}')
    cbar = plt.colorbar(shrink = 0.75)
    cbar.set_label('Biomass (Mg/ha)')
    plt.savefig(f'/home/s1949330/Documents/scratch/diss_data/model/{folder}/predict/{site}_{year}_PREDICT_AGB.png', dpi = 300)


# Code #############################################################################################################
if __name__ == '__main__':

    # Define command line arguments
    parser = argparse.ArgumentParser(description = "Extract data for a given site over given year(s).")
    parser.add_argument('--label', type = str, default = 'All', help = "Predictor label to find filtered (by correlation matrix) variables (e.g., Landsat, Sentinel, Palsar, All)")
    parser.add_argument("--model", type = int, required = True, choices = [1, 2, 3, 4, 5], help = "Name of trained model to use for predictions")
    parser.add_argument("--folder", type = str, required = True, help = "Directory containing model.")
    parser.add_argument("--site", type = str, required = True, help = "Study site by SEOSAW abbreviation.")
    parser.add_argument("--year", type = int, required = True, help = "End of austral year, eg: for Aug 2019 to July 2020, give 20 for 2020.")
    parser.add_argument("--batch", type = float, default = 0.02, help = 'Proportion of study site to compute predictions between 0-1, give 0.02 for 2 percent of data')
    args = parser.parse_args()

    # Identify predictor variables
    vars = glob(f'/home/s1949330/Documents/scratch/diss_data/pred_vars/{args.site}/{args.year}_*.tif')
    srtm_vars = glob(f'/home/s1949330/Documents/scratch/diss_data/pred_vars/{args.site}/SRTM_*.tif')
    all_vars = sorted(vars + srtm_vars)

    # Isolate filtered predictor variables
    pred_vars = filter_vars(all_vars, args.label)

    # Prepare predictor variables
    pred_flat = prepare_vars(pred_vars)
    print(pred_flat.shape)

    # Batch process model predictions
    model_dir = f'/home/s1949330/Documents/scratch/diss_data/model/{args.folder}/'
    pred_agb = predict_agb(pred_flat, args.batch, model_dir, args.model)

    # Sanity check biomass predictions 
    print('Max:', np.max(pred_agb))
    print('Mean:', np.mean(pred_agb))
    print('Median:', np.median(pred_agb))
    print('Min:', np.min(pred_agb))

    # Plot histogram of biomass predictions
    pred_hist(pred_agb, args.bins, args.site, args.year)

    # Visualise biomass prediction
    ref_var = GeoTiff(pred_vars[0])
    pred_map(pred_agb, ref_var, args.folder, args.site, args.year)

    # Save biomass map as geotiff
