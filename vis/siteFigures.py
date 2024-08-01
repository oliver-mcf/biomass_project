
# Visualise Data for Site Figures

# Libraries ########################################################################################################
from libraries import *


# Objects & Methods ################################################################################################
class GeoTiff:
    def __init__(self, filename):
        self.filename = filename
        self.read(filename)

    def read(self, filename):
        '''Read geotiff data'''
        # Read file metadata
        ds = gdal.Open(filename)
        proj = osr.SpatialReference(wkt = ds.GetProjection())
        self.crs = proj.GetAttrValue('AUTHORITY', 1)
        self.nX = ds.RasterXSize
        self.nY = ds.RasterYSize
        # Extract geolocation information
        transform_ds = ds.GetGeoTransform()
        self.xOrigin = transform_ds[0]
        self.yOrigin = transform_ds[3]
        self.pixelWidth = transform_ds[1]
        self.pixelHeight = transform_ds[5]
        # Store pixel data
        self.data = ds.GetRasterBand(1).ReadAsArray(0, 0, self.nX, self.nY)
        self.footprints = self.data[~np.isnan(self.data)]
        valid_indices = np.where(~np.isnan(self.data))
        self.x = self.xOrigin + valid_indices[1] * self.pixelWidth
        self.y = self.yOrigin + valid_indices[0] * self.pixelHeight

def plot_hist(data, bins, label, name):
    '''Plot Histogram from Array'''
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize = (8,6))
    data_flat = data.flatten()
    plt.hist(data_flat, bins = bins, color = 'teal', edgecolor = 'white', alpha = 0.6)
    plt.ylabel('Frequency')
    plt.xlabel(f'{label}')
    plt.savefig(f'/home/s1949330/scratch/diss_data/figures/study_sites/{name}.png', dpi = 300)
    plt.close()

def plot_bar(csv_file, site, label, name):
    '''Plot Bar Chart from Array'''
    df = pd.read_csv(csv_file)
    years = df.iloc[:, 0]
    data = df[f'{site}']
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize = (8,6))
    plt.bar(years, data, align = 'center', color = 'lightsteelblue', edgecolor = 'white')
    plt.ylabel(f'{label}')
    plt.savefig(f'/home/s1949330/scratch/diss_data/figures/study_sites/{name}.png', dpi = 300)
    plt.close()

def plot_line(csv_file, site, label, name):
    '''Plot Line Chart from Array'''
    df = pd.read_csv(csv_file)
    months = df.iloc[:, 0]
    data = df[f'{site}']
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize = (8,6))
    plt.plot(months, data, color = 'navy', marker = '.')
    plt.ylabel(f'{label}')
    plt.savefig(f'/home/s1949330/scratch/diss_data/figures/study_sites/{name}.png', dpi = 300)
    plt.close()

def plot_lines(csv_file1, csv_file2, site, name):
    '''Plot Line Chary with Two Axes from Arrays'''
    df1 = pd.read_csv(csv_file1)
    months1 = df1.iloc[:, 0]
    data1 = df1[f'{site}']
    df2 = pd.read_csv(csv_file2)
    months2 = df2.iloc[:, 0]
    data2 = df2[f'{site}']
    plt.rcParams['font.family'] = 'Arial'
    fig, ax1 = plt.subplots(figsize=(6, 6))
    ax1.bar(months1, data1, align = 'center', color = 'lightsteelblue', edgecolor = 'white')
    ax1.set_ylabel('Precipitation (mm)')
    ax1.set_ylim(0, 250)
    ax2 = ax1.twinx()
    ax2.plot(months2, data2, color='forestgreen', marker='.', linestyle='-', markersize = 10)
    ax2.set_ylabel('NDVI')
    ax2.set_ylim(0, 0.45)
    plt.savefig(f'/home/s1949330/scratch/diss_data/figures/study_sites/{name}.png', dpi=300)
    plt.close()

def plot_density(tif, grid_size, name):
    var = GeoTiff(tif)
    # Create a boolean mask where values > 0
    data = var.data
    # Count the number of points in each cell
    height = var.nY
    width = var.nX  # Get the dimensions of the data
    # Calculate the number of rows and columns in the density map
    n_rows = (height + grid_size - 1) // grid_size
    n_cols = (width + grid_size - 1) // grid_size
    # Initialize the density map with zeros
    density_map = np.zeros((n_rows, n_cols))
    # Pad the data to ensure that the grid covers the entire area
    padded_data = np.pad(data, ((0, grid_size - (height % grid_size if height % grid_size != 0 else grid_size)), 
                               (0, grid_size - (width % grid_size if width % grid_size != 0 else grid_size))),
                        mode='constant', constant_values=0)
    # Iterate over the grid cells
    for i in range(0, padded_data.shape[0], grid_size):
        for j in range(0, padded_data.shape[1], grid_size):
            # Extract the current grid cell
            cell_data = padded_data[i:i+grid_size, j:j+grid_size]
            # Count the number of points greater than 0 in the current grid cell
            count = np.sum(cell_data > 0)
            density_map[i // grid_size, j // grid_size] = count
    # Plot the density map
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.imshow(density_map, cmap = 'pink', interpolation = 'nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(label = 'Count', shrink = 0.5)
    plt.savefig(f'/home/s1949330/scratch/diss_data/figures/study_sites/{name}.png', dpi=300)
    plt.close()

def get_csv(tif_list, out_name):
    col_names = [os.path.splitext(os.path.basename(tif))[0] for tif in tif_list]
    flattened_data = []
    for tif in tif_list:
        var = GeoTiff(tif)  # Assuming GeoTiff is defined elsewhere
        #data = var.data[var.data > 0]  # Extract data where values are > 0
        data_flat = var.data.flatten()  # Flatten the 2D array into 1D
        flattened_data.append(data_flat)
    # Stack arrays as columns to get the shape (number of data points, number of TIFF files)
    flattened_array = np.column_stack(flattened_data)
    # Create DataFrame
    df = pd.DataFrame(flattened_array, columns=col_names)
    pprint(df.head())
    df.to_csv(f'/home/s1949330/scratch/diss_data/figures/{out_name}.csv', index = False)
    # Count the number of non-NaN and NaN values in each column
    for col in df.columns:
        count = df[col].notna().sum()
        print(f"{col} = {count}")


# Code #############################################################################################################
if __name__ == '__main__':

    # Define command line arguments
    parser = argparse.ArgumentParser(description = "Extract data for a given site over given year(s).")
    parser.add_argument("--site", required = True, help = "SEOSAW site (MGR or TKW)")
    #parser.add_argument("--label", required = True, help = "Units of predictor variable")
    parser.add_argument("--name", required = True, help = "Name of output file")
    args = parser.parse_args()

    # Read file
    tif = f'/home/s1949330/scratch/diss_data/pred_vars/{args.site}/SRTM_Elevation.tif'
    var = GeoTiff(tif)

    # Plot histogram
    plot_hist(var.data, 25, args.label, args.name)

    # Plot bar chart
    csv_file = '/home/s1949330/scratch/diss_data/figures/study_sites/annual_precip.csv'
    plot_bar(csv_file, args.site, args.label, args.name)

    # Plot line chart
    csv_file1 = '/home/s1949330/scratch/diss_data/figures/study_sites/month_precip.csv'
    csv_file2 = '/home/s1949330/scratch/diss_data/figures/study_sites/month_ndvi.csv'
    plot_lines(csv_file1, csv_file2, args.site, args.name)

    # Plot density map
    tif = f'/home/s1949330/scratch/diss_data/gedi/{args.site}/ALL_GEDI_AGB.tif'
    plot_density(tif, 20, args.name)

    # Store tif data in csv
    tif_list = glob(f'/home/s1949330/scratch/diss_data/gedi/{args.site}/2*_GEDI_AGB.tif')
    get_csv(tif_list, args.name)
