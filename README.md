# Biomass Project

## Overview

This library of code was developed for my dissertation, titled: "**In-Situ Validation Reveals Poor Performance of Extrapolated GEDI Aboveground Biomass Density Across Miombo Landscapes**". The dissertation contributed towards my MSc Earth Observation and Geoinformation Management at the University of Edinburgh.

The aim of my study was to assess the predictive performance of extrapolated GEDI Aboveground Biomass Density (AGBD) estimates across space and time using in-situ field AGBD estimates for validation in the miombo region of Southern Africa. For further detail, see `MSc_Dissertation.pdf`.

## Contents
- [Repository Structure](#Repository-Structure)
- [Script Functionality](#Script-Functionality)

## Repository Structure

**`gee/`**: JavaScript scripts to pre-process datasets in Google Earth Engine.

- `siteSeasonality.js`: Process Landsat NDVI and ERA-5 LAND Precipitation datasets for study sites.
- `gediBiomass.js`: Process annual composites of GEDI AGBD and Canopy Cover Fraction data.
- `landsatReflectance.js`: Process Landsat surface reflectance for band specific metrics.
- `landsatNDVI_median.js`: Process median annual Landsat Normalised Difference Vegetation Index (NDVI).
- `landsatNDVI_gradient.js`: Process annual Landsat NDVI change gradient from 5th to 95th percentiles.
- `landsatNDVI_precip.js`: Process annual Landsat NDVI percentiles with ERA-5 LAND Precipitation conditions.
- `sentinelCBackscatter.js`: Process annual Sentinel-1 Ground Range Detected C-Band Backscatter metrics.
- `palsarLBackscatter.js`: Process annual PALSAR-2 ScanSAR L-Band Backscatter metrics.
- `srtmTopography.js`: Process SRTM topographic metrics.
  
**`main/`**: Python and R source scripts used for main processing and computing. 

- `gediSeasonality.R`: Analyse seasonal artefacts in aggregated GEDI AGBD time series data.
- `libraries.py`: Import necessary python packages.
- `extractData.py`: Isolate intersecting GEDI AGBD footprints and predictor variable data.
- `alignData.py`: Merge input calibration data and perform geolocation filter.
- `filterFeatures.py`: Apply paired correlation coefficient filter.
- `assessData.py`: Perform linear regression and quantify GEDI AGBD footprints.
- `testModelTrees.py`: Examine model performance with various parameter configurations.
- `trainModel.py`: Calibrate and test group and site model conditions.
- `testModelSite.py`: Calibrate and test site model conditions with spatial cross calibration.
- `predictBiomass.py`: Extrapolate GEDI AGBD estimates across sites and years.
- `validateBiomass.py`: Perform validation of extrapolated GEDI AGBD estimates with in-situ data.

**`vis/`**: Python and R scripts used primarily for visualising outputs of the main scripts.

- `siteFigures.py`: Visualise average monthly precipitation and photosynthetic vibrance (NDVI).
- `visualPerformance.R`: Visualise model group performance and predicted-observed AGBD estimates.
- `visualValidation.R`: Visualise differentiated scatter and sensitivity plots for extrapolated-field AGBD estimates.


## Script Functionality

All `main/` and `vis/` were developed to address one or more of the following research questions:
1. How consistent is predictive performance when different EO predictor variables are used to extrapolate GEDI AGBD estimates across both sites?
2. How consistent is predictive performance of extrapolated GEDI AGBD estimates with spatial (site) cross validation?
3. How consistent is predictive performance of extrapolated GEDI AGBD estimates when validated with in-situ field AGBD estimates across both sites?


### Preparation

**gediSeasonality.R**

    .../src/main/ Rscript gediSeasonality.R
    
    >>> Monthly aggregated GEDI AGBD estimates for each austral year (2019-2023)
    >>> Temporal Auto-Correlation Function (ACF) of GEDI AGBD estimates
    >>> Auto-Regressive (AR) Model of GEDI AGBD estimates


**extractData.py**

    .../src/main/ python extractData.py --site TKW
                                        --site MGR
    
    >>> GEDI AGBD estimates and intersecting predictor variable data in .csv format for each combination of site and year.


**alignData.py**

    .../src/main/ python alignData.py --geo COVER
                                      --geo COVER --site TKW
                                      --geo COVER --site MGR
    
    >>> GEDI AGBD estimates and intersecting predictor variable data in .csv format for each site and overall, and for geolocation condition applied
    >>> Linear regression statistics in .csv format for all filtered predictor variables and GEDI AGBD estimates

**assessData.py**

    .../src/main/ python assessData.py --count --file ".../All_EXTRACT_MERGE.csv"
                                       --count --file ".../All_EXTRACT_MERGE_COVER.csv"
                                       --regress --file ".../All_EXTRACT_MERGE.csv"
                                       --regress --geo COVER --file ".../All_EXTRACT_MERGE_COVER"
    
    >>> Count of initial and filtered GEDI AGBD estimates for each site and year
    >>> Linear regression statistics for GEDI AGBD and predictor variables in .csv format

**filterFeatures.py**

    .../src/main/ python filterFeatures.py --filter --coef 0.9
                                           --filter --coef 0.9 --geo COVER
                                           --reduce
                                           --reduce --geo COVER
    
    >>> Final .csv files with filtered and reduced GEDI AGBD estimates for both and each site
    >>> Correlation coefficient matrix for all and geolocation filtered predictor variables in .png format


### Research Question 1

**testModelTrees.py**

    .../src/main/ python testModelTrees.py --label All
    
    >>> Performance statistics from 5 model iterations with 100-500 trees/estimtors    

**trainModel.py**

    .../src/main/ python trainModel.py --geo COVER --label All --folder All
                                                   --label Landsat --folder Landsat
                                                   --label Sentinel --folder Sentinel
                                                   --label Palsar --folder Palsar
    
    >>> Train-Test subsets in .csv format
    >>> Pred-Test subsets in .csv format
    >>> Random Forest Model in .joblib format
    >>> Variable Importances in .csv format
    >>> Combined list of performance statistics in .csv format

**visualPerformance.R**

    .../src/main/ Rscript visualPerformance.R
    
    >>> Box plots of performance statistics by model group in .png format
    >>> Scatter plot of predicted-observed values in .png format
    >>> Two-dimensional histogram of predicted-observed values in .png format

### Research Question 2

**trainModel.py**

    .../src/main/ python trainModel.py --geo COVER --label All --site TKW --folder TKW
                                                               --site MGR --folder MGR
    
    >>> Train-Test subsets in .csv format
    >>> Pred-Test subsets in .csv format
    >>> Random Forest Model in .joblib format
    >>> Variable Importances in .csv format
    >>> Combined list of performance statistics in .csv format

**testModelSite.py**

    .../src/main/ python testModelSite.py --geo COVER --label All --trainSite TKW --testSite MGR --folder TKW-MGR
                                                                  --trainSite MGR --testSite TKW --folder MGR-TKW
                                         
    >>> Pred-Test subsets in .csv format
    >>> Random Forest Model in .joblib format
    >>> Variable Importances in .csv format
    >>> Combined list of performance statistics in .csv format

**visualPerformance.R**

    .../src/main/ Rscript visualPerformance.R
    
    >>> Scatter plot of extrapolated GEDI AGB estimates (70%) and the withheld subset (30%) in .png format
    >>> Two-dimensional histogram of extrapolated GEDI AGB estimates (70%) and the withheld subset (30%) in .png format

### Research Question 3

**predictBiomass.py**

    .../src/main/ python predictBiomass.py --geo COVER --folder All --model 4 --site TKW 
                                                                              --site MGR
    
    >>> Map of extrapolated GEDI AGBD estimates in .png format for each site and year
    >>> Histogram of extrapolated GEDI AGBD estimates in .png format for each site and year
    >>> Geolocated raster of extrapolated GEDI AGBD estimates in .tif format for each site and year

**validateBiomass.py**

    .../src/main/ python validateBiomass.py
    
    >>> Scatter plots of extrapolated GEDI AGBD estimates and field AGBD estimates for each site, year, and overall in .png format
    >>> Perforance statistics of each site and year condition in .csv format

**visualValidation.R**

    .../src/main/ Rscript visualValidaiton.R
    
    >>> Scatter plot of extrapolated GEDI AGBD estimates and field AGBD estimates in .png format
    >>> Two-dimensional histogram of extrapolated GEDI AGBD estimates and field AGBD estimates in .png format
