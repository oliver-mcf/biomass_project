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

- `siteSeasonality.js`: Retrieves Landsat NDVI and ERA-5 LAND Precipitation datasets for study site figure.
- `gediBiomass.js`: Retrieves, filters, and produces annual composites of GEDI AGBD and Canopy Cover Fraction data.
- `landsatReflectance.js`: Retrieves and processes Landsat surface reflectance for band specific metrics.
- `landsatNDVI_median.js`: Retrieves and produces median annual Landsat Normalised Difference Vegetation Index (NDVI) for phenology metrics.
- `landsatNDVI_gradient.js`: Retrieves and produces annual Landsat NDVI gradient (5th to 95th percentile) for phenology metrics.
- `landsatNDVI_precip.js`: Retrieves, processes, and produces annual Landsat NDVI percentiles with ERA-5 LAND Precipitation conditions for phenology metrics.
- `sentinelCBackscatter.js`: Retrieves and produces annual Sentinel-1 Ground Range Detected C-Band Backscatter metrics.
- `palsarLBackscatter.js`: Retrieves and produces annual PALSAR-2 ScanSAR L-Band Backscatter metrics.
- `srtmTopography.js`: Retrieves and produces SRTM topographic metrics.
  
**`main/`**: Python and R source scripts used for main processing and computing. 

- `gediSeasonality.R`:
- `libraries.py`:
- `extractData.py`:
- `alignData.py`:
- `assessData.py`:
- `filterFeatures.py`:
- `testModelTrees.py`:
- `trainModel.py`:
- `testModelSite.py`:
- `predictBiomass.py`:
- `validateBiomass.py`:

**`vis/`**: Python and R scripts used primarily for visualising outputs of the main scripts.

- `siteFigures.py`:
- `visualPerformance.R`:
- `visualValidation.R`:


