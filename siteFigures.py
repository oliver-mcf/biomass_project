
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
    plt.hist(data_flat, bins = bins, color = 'lightgray', edgecolor = 'white')
    plt.ylabel('Frequency')
    plt.xlabel(f'{label}')
    plt.savefig(f'/home/s1949330/data/diss_data/figures/study_sites/{name}.png', dpi = 300)
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
    plt.savefig(f'/home/s1949330/data/diss_data/figures/study_sites/{name}.png', dpi = 300)
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
    plt.savefig(f'/home/s1949330/data/diss_data/figures/study_sites/{name}.png', dpi = 300)
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
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.bar(months1, data1, align = 'center', color = 'lightsteelblue', edgecolor = 'white')
    ax1.set_xlabel('Months')
    ax1.set_ylabel('Precipitation (mm)')
    ax2 = ax1.twinx()
    ax2.plot(months2, data2, color='green', marker='.', linestyle='-', markersize = 10)
    ax2.set_ylabel('NDVI')
    plt.savefig(f'/home/s1949330/data/diss_data/figures/study_sites/{name}.png', dpi=300)
    plt.close()



# Code #############################################################################################################
if __name__ == '__main__':

    # Define command line arguments
    parser = argparse.ArgumentParser(description = "Extract data for a given site over given year(s).")
    parser.add_argument("--site", required = True, help = "SEOSAW site (MGR or TKW)")
    #parser.add_argument("--label", required = True, help = "Units of predictor variable")
    parser.add_argument("--name", required = True, help = "Name of output file")
    args = parser.parse_args()

    # Read file
    pred_var = f'/home/s1949330/data/diss_data/pred_vars/{args.site}/SRTM_Elevation.tif'
    var = GeoTiff(pred_var)

    # Plot histogram
    plot_hist(var.data, 25, args.label, args.name)

    # Plot bar chart
    csv_file = '/home/s1949330/data/diss_data/figures/study_sites/annual_precip.csv'
    plot_bar(csv_file, args.site, args.label, args.name)

    # Plot line chart
    csv_file1 = '/home/s1949330/data/diss_data/figures/study_sites/month_precip.csv'
    csv_file2 = '/home/s1949330/data/diss_data/figures/study_sites/month_ndvi.csv'
    plot_lines(csv_file1, csv_file2, args.site, args.name)