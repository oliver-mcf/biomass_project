
# Validate Biomass Predictions with Field Data

# Libraries ########################################################################################################
from libraries import *
from extractData import GeoTiff

# Objects & Methods ################################################################################################

def read_plots(csv_file, year = None, site = None):
    '''Read Field AGB Estimates'''
    df = pd.read_csv(csv_file)
    if site:
        df = df[df['Plot'].str.startswith(site)]
    if year:
        field_biomass = df[f'AGB_{year}'] 
    lon = df['Lon']
    lat = df['Lat']
    x, y = reproject_plots(lon, lat, 3857)
    result_df = pd.DataFrame({'X': x, 'Y': y, f'AGB_{year}': field_biomass})
    result_df = result_df.dropna()
    return result_df

def reproject_plots(lon, lat, epsg):
    '''Reproject SEOSAW Plot Coordinates'''
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy = True)
    x, y = transformer.transform(lon.values, lat.values)
    return x, y

def intersect_plots(gedi, field_df):
    '''Identify Intersecting Plot-Pixel AGB estimates'''
    n_plots = len(field_df)
    plot_intersect = np.full(n_plots, np.nan, dtype = float)
    for i, plot in enumerate(range(n_plots)):
        xInd = np.floor((field_df['X'].iloc[i] - gedi.xOrigin) / gedi.pixelWidth).astype(int)
        yInd = np.floor((field_df['Y'].iloc[i] - gedi.yOrigin) / gedi.pixelHeight).astype(int)
        if 0 <= xInd < gedi.nX and 0 <= yInd < gedi.nY:
            pixel_value = gedi.data[yInd, xInd]
            plot_intersect[i] = pixel_value
    return plot_intersect


# compute statistical metrics

# plot scatter graph


# Code #############################################################################################################
if __name__ == '__main__':

    # Define command line arguments
    parser = argparse.ArgumentParser(description = 'Perform In-Situ Validation of GEDI AGB Estimates.')
    parser.add_argument('--year', help = 'Isolate SEOSAW data to given austral year: 17, 18, 21.')
    parser.add_argument('--site', type = str, help = 'SEOSAW study site: MGR or TKW.')
    args = parser.parse_args()

    # Read field AGB estimates
    csv_file = '/home/s1949330/data/diss_data/seosaw/seosaw_agb.csv'
    field_df = read_plots(csv_file, args.year, args.site)
    pprint(field_df)

    # Read GEDI AGB estimates
    pred = f'/home/s1949330/data/diss_data/model/yes_geo/All/predict/{args.site}_{args.year}_PREDICT_AGB_COVER.tif'
    gedi = GeoTiff(pred)

    # Isolate GEDI AGB estimates in plots
    plot_intersect = intersect_plots(gedi, field_df)
    print(plot_intersect)

    