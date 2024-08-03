# Biomass Project

## Overview

This library of code was developed for my dissertation, titled: "**In-Situ Validation Reveals Poor Performance of Extrapolated GEDI Aboveground Biomass Density Across Miombo Landscapes**". The dissertation contributed towards my MSc Earth Observation and Geoinformation Management at the University of Edinburgh.

The aim of my study was to assess the predictive performance of extrapolated GEDI Aboveground Biomass Density (AGBD) estimates across space and time using in-situ field AGBD estimates for validation in the miombo region of Southern Africa. For further detail, see `MSc_Dissertation.pdf`.

## Contents
- [Repository Structure](#Repository-Structure)
- [Script Functionality](#Script-Functionality)
- [Outputs](#Outputs)

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

Each of the main scripts were developed to address three research questions:
1. How consistent is predictive performance when different EO predictor variables are used to extrapolate GEDI AGBD estimates across both sites?
2. How consistent is predictive performance of extrapolated GEDI AGBD estimates with spatial (site) cross validation?
3. How consistent is predictive performance of extrapolated GEDI AGBD estimates when validated with in-situ field AGBD estimates across both sites?

To produce model-ready calibration data, 5 scripts were run: `gediSeasonality.R`, `extractData.py`, `alignData.py`, `filterFeatures.py`, and `assessData.py`. The following code segment illustrates the customisation functionality of each python script as to prepare for directly addressing each research question.

    .../src/main/ Rscript gediSeasonality.R

    >>> Monthly aggregated GEDI AGBD estimates for each austral year (2019-2023)
    >>> Temporal Auto-Correlation Function (ACF) of GEDI AGBD estimates
    >>> Auto-Regressive (AR) Model of GEDI AGBD estimates

 
    .../src/main/ python extractData.py --site TKW
                                        --site MGR

    >>> GEDI AGBD estimates and intersecting predictor variable data in .csv format for each combination of site and year.


    .../src/main/ python alignData.py --geo COVER
                                      --geo COVER --site TKW
                                      --geo COVER --site MGR

    >>> GEDI AGBD estimates and intersecting predictor variable data in .csv format for each site and overall, and for geolocation condition applied
    >>> Linear regression statistics in .csv format for all filtered predictor variables and GEDI AGBD estimates


    .../src/main/ python assessData.py --count --file ".../All_EXTRACT_MERGE.csv"
                                       --count --file ".../All_EXTRACT_MERGE_COVER.csv"
                                       --regress --file ".../All_EXTRACT_MERGE.csv"
                                       --regress --geo COVER --file ".../All_EXTRACT_MERGE_COVER"

    >>> Count of initial and filtered GEDI AGBD estimates for each site and year
    >>> Linear regression statistics for GEDI AGBD and predictor variables in .csv format
    

    .../src/main/ python filterFeatures.py --filter --coef 0.9
                                           --filter --coef 0.9 --geo COVER
                                           --reduce
                                           --reduce --geo COVER

    >>> Final .csv files with filtered and reduced GEDI AGBD estimates for both and each site
    >>> Correlation coefficient matrix for all and geolocation filtered predictor variables in .png format




