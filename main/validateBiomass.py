
# Validate Biomass Predictions with Field Data

# Libraries ########################################################################################################
from libraries import *
from extractData import GeoTiff

# Objects & Methods ################################################################################################

def read_plots(csv_file, site = None, year = None):
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
    field_df['GEDI_AGB'] = plot_intersect
    return field_df

def validation_stats(field, gedi, site, year):
    '''Statistically Validate GEDI AGB with field AGB'''
    residuals = gedi - field
    constant = sm.add_constant(field)
    white_test = sm.stats.diagnostic.het_white(residuals, constant)
    stats_dict = {
        'R2 (r2_score)': r2_score(field, gedi),
        'R2 (manual)': stats.pearsonr(field, gedi)[0] ** 2,
        'R2 (variance)': explained_variance_score(field, gedi),
        'Bias': np.sum(gedi - field) / len(field),
        'MAE': np.mean(np.abs(field - gedi)),
        'MAE%': np.mean(np.abs(field - gedi)) / np.mean(field) * 100,
        'RMSE': sqrt(mean_squared_error(field, gedi)),
        'RMSE%': sqrt(mean_squared_error(field, gedi)) / np.mean(field) * 100,
        'R': stats.pearsonr(field, gedi)[0],
        'LM': white_test[0],
        'F': white_test[2],
        'P': white_test[1]}
    valid_stats = []
    valid_stats.append(stats_dict)
    stats_df = pd.DataFrame(valid_stats)
    pprint(stats_df)
    stats_df.to_csv(f'/home/s1949330/scratch/diss_data/model/yes_geo/All/predict/validate/{site}_{year}_VALIDATION_STATS.csv', index = False)

def validation_plot(field, gedi, site, year):
    '''Plot Scatter of Observed and Predicted Values'''
    # Plot scatter
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize = (5,5))
    ax.scatter(field, gedi, marker = 'o', color = 'tab:blue')
    ax.plot([0, 120], [0, 120], ls = 'solid', color = 'k')
    # Plot line of best fit
    slope, intercept = np.polyfit(field, gedi, 1)
    best_fit = slope * np.array(field) + intercept
    ax.plot(field, best_fit, ls = 'solid', color = 'red', label = 'Linear')
    ax.set_xlim([0, 120])
    ax.set_ylim([0, 120])
    ax.set_xticks(np.arange(0, 120 + 10, 10))
    ax.set_yticks(np.arange(0, 120 + 10, 10))
    ax.set_xlabel('Field AGB Estimates (Mg/ha)')
    ax.set_ylabel('Extrapolated GEDI AGB Estimates (Mg/ha)')
    ax.set_title(f'Model Validation')
    fig_name = f'/home/s1949330/scratch/diss_data/model/yes_geo/All/predict/validate/{site}_{year}_VALIDATION_PLOT.png'
    plt.savefig(fig_name, dpi = 300)
    plt.close(fig)

# Code #############################################################################################################
if __name__ == '__main__':

    # Initialise validation
    validation = pd.DataFrame(columns = ["Site", "Year", "X", "Y", "GEDI_AGB", "Field_AGB"])
    year_list = ['17', '21']
    site_list = ['MGR', 'TKW']
    
    for site in site_list:
        for year in year_list:
            
            # Read field AGB estimates
            csv_file = '/home/s1949330/data/diss_data/seosaw/seosaw_agb.csv'
            field_df = read_plots(csv_file, site, year)

            # Read GEDI AGB estimates
            pred = f'/home/s1949330/scratch/diss_data/model/yes_geo/All/predict/{site}_{year}_PREDICT_AGB_COVER.tif'
            gedi_obj = GeoTiff(pred)

            # Isolate GEDI AGB estimates in plots
            intersect_df = intersect_plots(gedi_obj, field_df)

            # Calculate validation statistics
            gedi = intersect_df['GEDI_AGB']
            field = intersect_df[f'AGB_{year}']
            validation_stats(field, gedi, site, year)
            validation_plot(field, gedi, site, year)

            # Append intersecting data to the validation DataFrame
            for i in range(len(intersect_df)):
                validation = validation.append({
                    "Site": site,
                    "Year": year,
                    "X": intersect_df.iloc[i]['X'],
                    "Y": intersect_df.iloc[i]['Y'],
                    "GEDI_AGB": intersect_df.iloc[i]['GEDI_AGB'],
                    "Field_AGB": field.iloc[i]
                }, ignore_index = True)

    # Save combined data to CSV
    pprint(validation)
    validation.to_csv('/home/s1949330/scratch/diss_data/model/yes_geo/All/predict/validate/BOTH_ALL_VALIDATION.csv', index = False)

    # Calculate statistics for all validation data
    all_gedi = validation['GEDI_AGB']
    all_field = validation['Field_AGB']
    validation_stats(all_field, all_gedi, 'BOTH', 'ALL')
    validation_plot(all_field, all_gedi, 'BOTH', 'ALL')

    # Calculate statistics for site validation data
    plot_sites = validation['Site'].unique()
    for abb in plot_sites:
        filtered_df = validation.loc[validation['Site'] == abb, ['Field_AGB', 'GEDI_AGB']]
        all_field = filtered_df['Field_AGB']
        all_gedi = filtered_df['GEDI_AGB']
        validation_stats(all_field, all_gedi, abb, 'ALL')
        validation_plot(all_field, all_gedi, abb, 'ALL')
    
