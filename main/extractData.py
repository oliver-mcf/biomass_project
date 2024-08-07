
# Extract Data for Model Input

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

def get_epsg(dataset):
    proj = dataset.GetProjection()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj)
    epsg = srs.GetAttrValue('AUTHORITY', 1)
    return epsg

def reproject(file_list, epsg):
    '''Reproject geotiff files with EPSG code'''
    for file in tqdm(file_list, desc = 'REPROJECT', unit = 'file'):
        ds = gdal.Open(file)
        # Identify files to reproject
        init_epsg = get_epsg(ds)
        if init_epsg != epsg:
            output_file = file
            # Perform reprojection
            ds_reproj = gdal.Warp(output_file, ds, dstSRS = f"EPSG:{epsg}", format = 'GTiff')
            ds_reproj = None
        ds = None

def get_res(file):
    '''Retrieve geotiff pixel resolution'''
    ds = gdal.Open(file)
    res = ds.GetGeoTransform()[1]
    ds = None
    return res

def resample(file_list, res):
    '''Resample geotiff files to set resolution'''
    for file in tqdm(file_list, desc = 'RESAMPLE', unit = 'file'):
        ds = gdal.Open(file)
        # Identify files to resample
        init_res = ds.GetGeoTransform()[1]
        if init_res != res:
            output_file = file
            ds_res = gdal.Warp(output_file, ds, xRes = res, yRes = res, resampleAlg = 'bilinear', format = 'GTiff')  
            ds_res = None
        ds = None

def intersect(gedi, file):
    '''Isolate GEDI footprint indices in predictor variables'''
    var = GeoTiff(file)
    nGedi = len(gedi.footprints)
    var_intersect = np.full(nGedi, np.nan, dtype = float)
    for j in range(nGedi):
        # Isolate pixel indices of target data
        x = gedi.x[j]
        y = gedi.y[j]
        # Calculate corresponding pixel indices of predictor data
        xInd = np.floor((x - var.xOrigin) / var.pixelWidth).astype(int)
        yInd = np.floor((y - var.yOrigin) / var.pixelHeight).astype(int)
        if 0 <= xInd < var.nX and 0 <= yInd < var.nY:
            pixel_value = var.data[yInd, xInd]
            var_intersect[j] = pixel_value
    return var_intersect

def extract(gedi, pred_vars):
    '''Extract GEDI footprints and intersecting predictor variable data'''
    # Extract intersecting pixels in predictor data
    all_intersects = []
    for var in tqdm(pred_vars, desc = "EXTRACT"):
        intersecting_data = intersect(gedi, var)
        all_intersects.append(intersecting_data)
    # Align target data and predictor data
    all_intersects = np.array(all_intersects).T
    gedi_coordinates = np.column_stack((gedi.x, gedi.y))
    gedi_values = gedi.footprints.reshape((-1, 1))
    extracted_data = np.hstack((gedi_coordinates, gedi_values, all_intersects))
    return extracted_data

def export(array, labels, site, year):
    '''Save GEDI and predictor variable data as CSV'''
    # Set predictor variable names
    var_names = [os.path.splitext(os.path.basename(label))[0] for label in labels]
    column_names = ['GEDI_X', 'GEDI_Y', 'GEDI_AGB'] + var_names
    df = pd.DataFrame(array, columns = column_names)
    df_sorted = df.sort_values(by = 'GEDI_AGB').reset_index(drop = True)
    print(df_sorted.head())
    print(df_sorted.shape)
    output = f'/home/s1949330/scratch/diss_data/pred_vars/input_init/{site}_{year}_INPUT_DATA.csv'
    df_sorted.to_csv(output, index = False)
    print(f"Successful export: {output}\n")


# Code #############################################################################################################
if __name__ == '__main__':

    # Define command line arguments
    parser = argparse.ArgumentParser(description = "Extract data for a given site over given year(s).")
    parser.add_argument("--site", required = True, help = "Study site by SEOSAW abbreviation.")
    args = parser.parse_args()

    # Define calibration period
    year_list = ['20','21','22','23']
    for year in year_list:
        print(f'Extracting data in: 20{year}')

        # Read GEDI data
        input_var = f'/home/s1949330/scratch/diss_data/gedi/{args.site}/{year}_GEDI_AGB.tif'
        gedi = GeoTiff(input_var)

        # Read predictor variables
        cover_var = glob(f'/home/s1949330/scratch/diss_data/gedi/{args.site}/{year}_GEDI_COVER.tif')
        vars = glob(f'/home/s1949330/scratch/diss_data/pred_vars/{args.site}/{year}_*.tif')
        srtm_vars = glob(f'/home/s1949330/scratch/diss_data/pred_vars/{args.site}/SRTM_*.tif')
        pred_vars = sorted(cover_var + vars + srtm_vars)

        # Reproject all variables
        var_list = [input_var] + pred_vars
        reproject(var_list, epsg = '3857')

        # Resample predictor variables to GEDI resolution
        gedi_res = get_res(input_var)
        resample(pred_vars, gedi_res)

        # Extract GEDI footprints and intersecting pixels
        extracted_data = extract(gedi, pred_vars)

        # Export labelled variables to csv
        export(extracted_data, pred_vars, args.site, year)
