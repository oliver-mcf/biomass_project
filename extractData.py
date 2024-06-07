
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


# Objects & Methods ################################################################################################
class GeoTiff:
    def __init__(self, filename):
        self.filename = filename
        self.read(filename)

    def read(self, filename):
        '''Read geotiff data'''
        ds = gdal.Open(filename)
        proj = osr.SpatialReference(wkt = ds.GetProjection())
        self.crs = proj.GetAttrValue('AUTHORITY',1)
        self.nX = ds.RasterXSize             # number of pixels in x direction
        self.nY = ds.RasterYSize             # number of pixels in y direction
        transform_ds = ds.GetGeoTransform()  # extract geolocation information
        self.xOrigin = transform_ds[0]       # coordinate of x corner
        self.yOrigin = transform_ds[3]       # coordinate of y corner
        self.pixelWidth = transform_ds[1]    # resolution in x direction
        self.pixelHeight = transform_ds[5]   # resolution in y direction
        self.data = ds.GetRasterBand(1).ReadAsArray(0, 0, self.nX, self.nY)
        self.data[np.isnan(self.data)] = -999
        self.data_valid = self.data[self.data != -999]
        self.x = np.linspace(self.xOrigin, self.xOrigin + self.pixelWidth * self.nX, self.nX)
        self.y = np.linspace(self.yOrigin, self.yOrigin + self.pixelHeight * self.nY, self.nY)
        
        print('Success reading:', filename)




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
    pprint(gedi.data.shape)
    pprint(gedi.pixelWidth)
    pprint(gedi.data_valid)

    # Read Landsat data
    pred_vars = [f'/home/s1949330/Documents/scratch/diss_data/pred_vars/{args.site}/{args.year}_NDVI_Dry95.tif',
                 f'/home/s1949330/Documents/scratch/diss_data/pred_vars/{args.site}/{args.year}_NDVI_Dry05.tif']
    n = len(pred_vars)
    sent = np.empty((n), dtype = GeoTiff)
    for i, filename in enumerate(pred_vars):
        sent[i] = GeoTiff(filename)

    # Allocate space for intersecting GEDI-Landsat data
    nGedi = len(gedi.data_valid.flatten())
    nSent = sent.shape[0]
    intersect = np.full((nGedi, nSent), -999, dtype=float)

    # Index intersecting pixels
    xInd = np.array(np.floor((gedi.x - sent[0].xOrigin) / sent[0].pixelWidth), dtype=int)
    xInd -= np.min(xInd)
    yInd = np.array(np.floor((gedi.y - sent[0].yOrigin) / sent[0].pixelHeight), dtype=int)
    yInd -= np.min(yInd)
    print("Adjusted xInd:", xInd)
    print("Adjusted yInd:", yInd)

    # Loop over GEDI footprints
    for j in range(min(nGedi, len(xInd))):
        if 0 <= xInd[j] < sent[0].nX and 0 <= yInd[j] < sent[0].nY:
            # Loop over Landsat variables
            for i in range(nSent):
                intersect[j, i] = sent[i].data[yInd[j], xInd[j]]

    print(intersect.shape)
    print(len(intersect[intersect != -999]))
