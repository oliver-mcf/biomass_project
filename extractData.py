
# Extract Data for Model Input

# Libraries ########################################################################################################
import numpy as np
from pyproj import Proj, transform
from osgeo import gdal, osr
from math import floor
from glob import glob
from tqdm import tqdm
import argparse
from pprint import pprint
import os
import pandas as pd


# Objects & Methods ################################################################################################
class GeoTiff:
    def __init__(self, filename):
        self.filename = filename
        self.read(filename)

    def read(self, filename):
        '''Read geotiff data'''
        ds = gdal.Open(filename)
        proj = osr.SpatialReference(wkt = ds.GetProjection())
        self.crs = proj.GetAttrValue('AUTHORITY', 1)
        self.nX = ds.RasterXSize             # number of pixels in x direction
        self.nY = ds.RasterYSize             # number of pixels in y direction
        transform_ds = ds.GetGeoTransform()  # extract geolocation information
        self.xOrigin = transform_ds[0]       # coordinate of x corner
        self.yOrigin = transform_ds[3]       # coordinate of y corner
        self.pixelWidth = transform_ds[1]    # resolution in x direction
        self.pixelHeight = transform_ds[5]   # resolution in y direction
        self.data = ds.GetRasterBand(1).ReadAsArray(0, 0, self.nX, self.nY)
        self.footprints = self.data[~np.isnan(self.data)]
        valid_indices = np.where(~np.isnan(self.data))
        self.x = self.xOrigin + valid_indices[1] * self.pixelWidth
        self.y = self.yOrigin + valid_indices[0] * self.pixelHeight

def intersect(gedi, landsat_filename):
    '''Isolate GEDI footprint indices in Landsat data'''
    landsat = GeoTiff(landsat_filename)
    nGedi = gedi.footprints.shape[0]
    landsat_intersect = np.full(nGedi, -999, dtype=float)
    for j in range(nGedi):
        x = gedi.x[j]
        y = gedi.y[j]
        xInd = np.floor((x - landsat.xOrigin) / landsat.pixelWidth).astype(int)
        yInd = np.floor((y - landsat.yOrigin) / landsat.pixelHeight).astype(int)
        if 0 <= xInd < landsat.nX and 0 <= yInd < landsat.nY:
            landsat_intersect[j] = landsat.data[yInd, xInd]
    return landsat_intersect

def extract(gedi, pred_vars):
    '''Extract GEDI footprints and intersecting predictor variable data'''
    all_intersects = []
    for var in tqdm(pred_vars, desc = "Processing predictor variables..."):
        intersecting_data = intersect(gedi, var)
        all_intersects.append(intersecting_data)
    all_intersects = np.array(all_intersects).T
    gedi_coordinates = np.column_stack((gedi.x, gedi.y))
    gedi_values = gedi.footprints.reshape((-1, 1))
    extracted_data = np.hstack((gedi_coordinates, gedi_values, all_intersects))
    extracted_data[extracted_data == 0] = np.nan
    return extracted_data

def export(array, labels, site, year):
    var_names = [os.path.splitext(os.path.basename(label))[0] for label in labels]
    column_names = ['GEDI_X', 'GEDI_Y', 'GEDI_AGB'] + var_names
    df = pd.DataFrame(array, columns = column_names)
    pprint(df.head())
    print(df.shape)
    df.to_csv(f'/home/s1949330/Documents/scratch/diss_data/{site}_{year}_INPUT_DATA.csv', index = False)
    print(f"Successful export: /scratch/diss_data/{site}_{year}_INPUT_DATA.csv\n")


# Code #############################################################################################################
if __name__ == '__main__':

    # Define command line arguments
    parser = argparse.ArgumentParser(description = "Extract data for a given site over given year(s).")
    parser.add_argument("site", help = "Study site by SEOSAW abbreviation.")
    parser.add_argument("--year", help = "End of austral year, eg: for Aug 2019 to July 2020, give 20 for 2020.")
    args = parser.parse_args()

    # Read GEDI data
    input_var = f'/home/s1949330/Documents/scratch/diss_data/gedi/{args.site}/{args.year}_GEDI_AGB.tif'
    gedi = GeoTiff(input_var)

    # Read Landsat data
    landsat_vars = glob(f'/home/s1949330/Documents/scratch/diss_data/pred_vars/{args.site}/{args.year}_*.tif')
    srtm_vars = glob(f'/home/s1949330/Documents/scratch/diss_data/pred_vars/{args.site}/SRTM_*.tif')
    pred_vars = sorted(landsat_vars + srtm_vars)

    # Extract GEDI footprints and intersecting Landsat pixels
    extracted_data = extract(gedi, pred_vars)

    # Export labelled variables to csv
    export(extracted_data, pred_vars, args.site, args.year)