
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
        proj = osr.SpatialReference(wkt=ds.GetProjection())
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
        
        # Create valid x and y coordinates arrays
        valid_indices = np.where(~np.isnan(self.data))
        self.x = self.xOrigin + valid_indices[1] * self.pixelWidth
        self.y = self.yOrigin + valid_indices[0] * self.pixelHeight
        print('Success reading:', filename)

    def set_resolution(self, ref_geotiff):
        '''Set resolution to match the reference GeoTIFF'''
        self.pixelWidth = ref_geotiff.pixelWidth
        self.pixelHeight = ref_geotiff.pixelHeight
        print(f'Success setting resolution to match {ref_geotiff.pixelWidth}')

def read_list(pred_list):
    '''Read a list of geotiff files'''
    n = len(pred_list)
    pred_vars = np.empty((n), dtype=object)
    for i, filename in enumerate(tqdm(pred_list, desc="Reading Predictor Variables...")):
        pred_vars[i] = GeoTiff(filename)
        print(f'Success reading {i+1} of {n} variables.')
    return pred_vars

def intersect(gedi, landsat):
    '''Derive intersecting predictor variables with GEDI footprints'''
    # Allocate space for intersecting GEDI-Landsat data
    nGedi = gedi.footprints.shape[0]
    nLandsat = len(landsat)
    landsat_intersect = np.full((nGedi, nLandsat), -999, dtype=float)

    # Loop over GEDI footprints
    for j in range(nGedi):
        # Calculate the x and y indices for each GEDI footprint
        x = gedi.x[j]
        y = gedi.y[j]
        xInd = np.floor((x - landsat[0].xOrigin) / landsat[0].pixelWidth).astype(int)
        yInd = np.floor((y - landsat[0].yOrigin) / landsat[0].pixelHeight).astype(int)
        if 0 <= xInd < landsat[0].nX and 0 <= yInd < landsat[0].nY:
            # Loop over Landsat variables
            for i in range(nLandsat):
                landsat_intersect[j, i] = landsat[i].data[yInd, xInd]
                
    return landsat_intersect

# Code #############################################################################################################
if __name__ == '__main__':
    # Define command line arguments
    parser = argparse.ArgumentParser(description="Extract data for a given site over given year(s).")
    parser.add_argument("site", help="Study site by SEOSAW abbreviation.")
    parser.add_argument("--year", help="End of austral year, eg: for Aug 2019 to July 2020, give 20 for 2020.")
    args = parser.parse_args()

    # Read GEDI data
    input_var = f'/home/s1949330/Documents/scratch/diss_data/gedi/{args.site}/{args.year}_GEDI_AGB.tif'
    gedi = GeoTiff(input_var)

    # Read Landsat data
    pred_vars = [
        f'/home/s1949330/Documents/scratch/diss_data/pred_vars/{args.site}/{args.year}_NDVI_Dry95.tif',
        f'/home/s1949330/Documents/scratch/diss_data/pred_vars/{args.site}/{args.year}_NDVI_Dry05.tif'
    ]
    landsat = read_list(pred_vars)

    # Set the GEDI resolution to match the first Landsat file
    gedi.set_resolution(landsat[0])

    # Isolate Landsat pixels intersecting with GEDI footprints
    landsat_intersect = intersect(gedi, landsat)




    print('Shape of GEDI footprint data:', gedi.footprints.shape)
    print('Shape of intersecting Landsat data:', landsat_intersect.shape)
    print('')
    print('GEDI pixelWidth:', gedi.pixelWidth)
    print('Landsat pixelWidth:', landsat[0].pixelWidth)
    print('GEDI footprint biomass data:', gedi.footprints[:5])
    print('Intersecting Landsat data:', landsat_intersect)
    print('Shape of intersecting Landsat data:', landsat_intersect.shape)
    print('Intersecting pixels with Landsat data:', landsat_intersect[landsat_intersect != -999].shape)
    print('Number of GEDI footprints:', len(gedi.footprints.flatten()))
