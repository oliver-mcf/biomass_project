
# Predict AGB with Trained Model

# Libraries ########################################################################################################
from libraries import *
from extractData import GeoTiff


# Objects & Methods ################################################################################################
def resample(reference_path, geotiff_list):
    # Read reference GeoTIFF
    with rio.open(reference_path) as ref_dataset:
        ref_transform = ref_dataset.transform
        ref_crs = ref_dataset.crs
        ref_res = (ref_transform.a, -ref_transform.e)
    # Iterate over variable list
    for idx in tqdm(range(len(geotiff_list)), desc = 'Resampling predictor variables...'):
        geotiff_path = geotiff_list[idx]
        with rio.open(geotiff_path) as src_dataset:
            src_transform = src_dataset.transform
            src_crs = src_dataset.crs
            # Calculate dimensions for the resampled data
            transform, width, height = calculate_default_transform(
                src_crs, ref_crs, src_dataset.width, src_dataset.height, *src_dataset.bounds, resolution = ref_res)
            # Update profile to match the reference
            profile = src_dataset.profile
            profile.update({'crs': ref_crs, 'transform': transform, 'width': width, 'height': height})
            # Save resampled data
            with rio.open(geotiff_path, 'w', **profile) as dst_dataset:
                for i in range(1, src_dataset.count + 1):
                    reproject(source = rio.band(src_dataset, i),
                              destination = rio.band(dst_dataset, i),
                              src_transform = src_transform,
                              src_crs = src_crs,
                              dst_transform = transform,
                              dst_crs = ref_crs,
                              resampling = Resampling.bilinear)
    print('Resampling complete')

def prepare(site, year, batch_proportion=0.1):
    vars_files = glob(f'/home/s1949330/Documents/scratch/diss_data/pred_vars/{site}/{year}_*.tif')
    nVars = len(vars_files)
    pred_vars = []
    for var_file in vars_files:
        with rio.open(var_file) as dataset:
            pred_vars.append(dataset.read(1))
    nY, nX = pred_vars[0].shape
    pred_vars_flat = np.empty((nX * nY, nVars), dtype=float)
    batch_size = int(nX * nY * batch_proportion)
    print('Flattening predictor variables...')
    with tqdm(total=nX * nY) as progress_bar:
        for i in range(0, nX * nY, batch_size):
            chunk_end = min(i + batch_size, nX * nY)
            for j in range(nVars):
                band_data = pred_vars[j].flatten()
                if chunk_end > len(band_data):
                    chunk_end = len(band_data)
                pred_vars_flat[i:chunk_end, j] = band_data[i:chunk_end]
            progress_bar.update(chunk_end - i)
    print('Predictor variables flattened')
    return pred_vars_flat

def predict(pred_vars_flat, model_path, no_data_value=-999, batch_proportion=0.1):
    rf = joblib.load(model_path)
    n_samples, n_features = pred_vars_flat.shape
    print(n_features)
    batch_size = int(n_samples * batch_proportion)
    predictions = np.empty((n_samples,), dtype=float)
    print('Predicting AGB...')
    with tqdm(total=n_samples) as progress_bar:
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_data = pred_vars_flat[i:batch_end, :]
            mask = np.any(batch_data == no_data_value, axis=1)
            batch_data = batch_data[~mask]
            batch_data = np.nan_to_num(batch_data, nan=0.0, posinf=1e9, neginf=-1e9)
            batch_pred = rf.predict(batch_data)
            predictions[i:batch_end][~mask] = batch_pred
            progress_bar.update(batch_end - i)
    print('Successfully predicted AGB')
    print(f'Max: {np.max(predictions)} Mg/ha')
    print(f'Mean: {np.mean(predictions)} Mg/ha')
    print(f'Median: {np.median(predictions)} Mg/ha')
    print(f'Min: {np.min(predictions)} Mg/ha')
    return predictions

def map_biomass(site, year, predictions, reference, cmap = 'Greens'):
    BiomassMap = np.reshape(predictions, (reference.height, reference.width))
    plt.figure(figsize = (12, 8))
    plt.imshow(BiomassMap, cmap = cmap)
    plt.title(f'Predicted Aboveground Biomass: {site}, 20{year}')
    cbar = plt.colorbar(shrink = 0.75)
    cbar.set_label('AGB (Mg/ha)')
    plt.show()


# Code #############################################################################################################
if __name__ == '__main__':

     # Define command line arguments
    parser = argparse.ArgumentParser(description = "Extract data for a given site over given year(s).")
    parser.add_argument("site", help = "Study site by SEOSAW abbreviation.")
    parser.add_argument("--year", help = "End of austral year, eg: for Aug 2019 to July 2020, give 20 for 2020.")
    parser.add_argument("--model", help = "Specific trained model to predict AGB, eg: Landsat_FilteredMonths")
    args = parser.parse_args()

    # Resample data to match resolutions
    #reference_path = f'/home/s1949330/Documents/scratch/diss_data/pred_vars/REFERENCE_{args.site}_23_SR_B2_Median.tif'
    #geotiff_list = glob(f'/home/s1949330/Documents/scratch/diss_data/pred_vars/{args.site}/{args.year}*.tif')
    #resample(reference_path, geotiff_list)


    














    # Read and flatten predictor variables
    #pred_vars_flat = prepare(args.site, args.year)

    # Predict AGB across study area
    #model = f'/home/s1949330/Documents/scratch/diss_data/model/{args.model}/RF_MODEL.joblib'
    #predictions = predict(pred_vars_flat, model)

    # Map predicted AGB for study area
    #with rio.open(reference_path) as reference:
    #    map_biomass(args.site, args.year, predictions, reference)

    # Save map as geotiff