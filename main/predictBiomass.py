
# Predict Biomass with RF Model

# Libraries ########################################################################################################
from libraries import *


# Objects & Methods ################################################################################################
from extractData import GeoTiff, get_epsg, reproject, resample

def filter_vars(site, year, geo):
    '''Filter Variables to Match Model Predictors'''
    # Identify variable names used in model
    if geo:
        ref_csv = f'/home/s1949330/scratch/diss_data/pred_vars/input_final/All_FILTERED_VARIABLES_{geo}.csv'
    else:
        ref_csv = '/home/s1949330/scratch/diss_data/pred_vars/input_final/All_FILTERED_VARIABLES.csv'
    df = pd.read_csv(ref_csv)
    pred_vars = []
    for name in df['Retained']:
        if name.startswith("SRTM"):
            srtm_var = f'/home/s1949330/scratch/diss_data/pred_vars/{site}/{name}.tif'
            pred_vars.append(srtm_var)
        else:
            year_var = f'/home/s1949330/scratch/diss_data/pred_vars/{site}/{year}_{name}.tif'
            pred_vars.append(year_var)
    return pred_vars

def prepare_vars(pred_vars, year, site):
    '''Prepare Data for Model Predictions'''
    reproject(pred_vars, 3857)
    resample(pred_vars, 25)
    # Set common dimensions for variables
    one_var = f'/home/s1949330/scratch/diss_data/pred_vars/{site}/{year}_HV_Median.tif'
    ref_var = GeoTiff(one_var)
    if site == 'MGR':
        common_nX, common_nY = ref_var.nX, ref_var.nY
    elif site == 'TKW':
        common_nX, common_nY = ref_var.nX, (ref_var.nY - 2)
    print(f'Common Variable Dimensions: {common_nX} x {common_nY}') 
    # Iterate through predictor variables to read data
    flat_dataset = []
    for var in tqdm(pred_vars, desc = "PREPARE"):
        pred = GeoTiff(var)
        data = pred.data[:common_nY, :common_nX]
        data_flat = data.flatten()
        data_flat[np.isnan(data_flat)] = -999
        flat_dataset.append(data_flat)
    # Align flattened predictor variables
    pred_flat = np.stack(flat_dataset, axis = -1)
    print(pred_flat.shape)
    return pred_flat, common_nX, common_nY, ref_var

def predict_agb(pred_flat, batch, folder, model):
    # Configure batch process
    rf = joblib.load(f'/home/s1949330/scratch/diss_data/model/{folder}/All_RF_MODEL_FOLD{model}.joblib')
    pred_agb = np.empty((pred_flat.shape[0],), dtype = float)
    batch_size = int(batch * pred_flat.shape[0])
    num_batches = int(math.ceil(pred_flat.shape[0] / batch_size))
    # Predict in batches
    for i in tqdm(range(num_batches), desc = "PREDICT"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, pred_flat.shape[0])
        pred_agb[start_idx:end_idx] = rf.predict(pred_flat[start_idx:end_idx])
    return pred_agb

def pred_hist(pred_data, bins, site, year, folder, geo):
    '''Plot Histogram from Array'''
    # Mask data for sensitivity
    pred_data[pred_data > 120] = 120
    # Plot the histogram
    plt.rcParams['font.family'] = 'Arial'
    plt.figure(figsize = (8, 8))
    plt.hist(pred_data, bins = bins, color = 'teal', edgecolor = 'w')
    plt.title(f'Histogram of Extrapolated GEDI AGB Estimates for {site}, 20{year}')
    plt.ylabel('Frequency')
    max_val = np.max(pred_data)
    plt.xlim(0, max_val)
    plt.xticks(np.arange(0, max_val + 10, 10))
    plt.xlabel('Biomass (Mg/ha)')
    plt.savefig(f'/home/s1949330/scratch/diss_data/model/{folder}/predict/{site}_{year}_PREDICT_AGB_{geo}_HIST.png', dpi = 300)
    plt.close()

def pred_map(pred_data, nX, nY, folder, site, year, geo):
    '''Reshape Array of Predictions to Produce Map'''
    # Mask data for sensitivity
    pred_data[pred_data > 120] = 120
    # Visualise predicted biomass map
    plt.rcParams['font.family'] = 'Arial'
    agb_map = np.reshape(pred_data, (nY, nX))
    plt.figure(figsize = (10, 8))
    plt.imshow(agb_map, cmap = 'Greens')    
    plt.title(f'Extrapolated GEDI AGB Estimates for {site}, 20{year}')
    cbar = plt.colorbar(shrink = 0.5)
    cbar.set_label('Biomass (Mg/ha)')
    max_val = np.max(pred_data)
    cbar.set_ticks(np.arange(0, max_val + 1, 10))
    plt.savefig(f'/home/s1949330/scratch/diss_data/model/{folder}/predict/{site}_{year}_PREDICT_AGB_{geo}_MAP.png', dpi = 300)
    plt.close()
    return agb_map

def write_tif(pred_data, x_origin, pixel_width, y_origin, pixel_height, epsg, output_tif):
    geotransform = (x_origin, pixel_width, 0, y_origin, 0, pixel_height)    
    dst_ds = gdal.GetDriverByName('GTiff').Create(output_tif, nX, nY, 1, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg) 
    dst_ds.SetProjection(srs.ExportToWkt())
    dst_ds.GetRasterBand(1).WriteArray(pred_data)
    dst_ds.GetRasterBand(1).SetNoDataValue(-999)
    dst_ds.FlushCache()     
    dst_ds = None
    print("Written to ", output_tif)
    return


# Code #############################################################################################################
if __name__ == '__main__':

    # Define command line arguments
    parser = argparse.ArgumentParser(description = "Extract data for a given site over given year(s).")
    parser.add_argument("--model", type = int, required = True, help = "Name of trained model (1-10) to use for predictions")
    parser.add_argument("--folder", type = str, required = True, help = "Directory containing model.")
    parser.add_argument("--site", type = str, required = True, help = "Study site by SEOSAW abbreviation.")
    parser.add_argument("--batch", type = float, default = 0.05, help = 'Proportion of study site to compute predictions between 0-1 g: 0.05 for 5 percent')
    parser.add_argument("--bins", type = int, default = 20, help = 'Number of bins to arrange data in histogram')
    parser.add_argument("--geo", help = 'Geolocation filtered csv for retained variables')
    args = parser.parse_args()

    # Initialise years 
    year_list = ['17', '18', '19']
    for year in year_list:
        print(f'Predicting Biomass for 20{year}...')

        # Isolate filtered predictor variables
        pred_vars = filter_vars(args.site, year, args.geo)

        # Prepare predictor variables
        pred_flat, nX, nY, ref_var = prepare_vars(pred_vars, year, args.site)
        print(pred_flat.shape)

        # Batch process model predictions and store data as csv
        pred_agb = predict_agb(pred_flat, args.batch, args.folder, args.model)

        # Sanity check biomass predictions 
        print('Max:', np.max(pred_agb))
        print('Mean:', np.mean(pred_agb))
        print('Median:', np.median(pred_agb))
        print('Min:', np.min(pred_agb))

        # Plot histogram of biomass predictions
        pred_hist(pred_agb, args.bins, args.site, year, args.folder, args.geo)

        # Visualise biomass prediction
        agb_map = pred_map(pred_agb, nX, nY, args.folder, args.site, year, args.geo)

        # Save biomass map as geotiff
        output_tif = f'/home/s1949330/scratch/diss_data/model/{args.folder}/predict/{args.site}_{year}_PREDICT_AGB_{args.geo}.tif'
        write_tif(agb_map, ref_var.xOrigin, ref_var.pixelWidth, ref_var.yOrigin, ref_var.pixelHeight, 3857, output_tif)
