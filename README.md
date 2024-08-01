# Biomass Project

## Overview

This library of code was developed for my dissertation, titled: "**In-Situ Validation of Extrapolated GEDI Aboveground Biomass Reveals Poor Transferabilty Across Miombo Landscapes**". The dissertation contributed towards my MSc Earth Observation and Geoinformation Management at the University of Edinburgh.

The aim of my study was to assess the predictive performance of extrapolated GEDI Aboveground Biomass (AGB) estimates across space and time using in-situ field AGB estimates for validation in the miombo region of Southern Africa. For further detail, see `MSc_Dissertation.pdf`.

## Contents
- [Repository Structure](#Repository-Structure)
- [Script Functionality](#Script-Functionality)
- [Outputs](#Outputs)

## Repository Structure

**`gee/`**: JavaScript scripts to pre-process datasets in Google Earth Engine.

- `siteSeasonality.js`:
- `gediBiomass.js`:
- `landsatReflectance.js`:
- `landsatNDVI_median.js`:
- `landsatNDVI_gradient.js`:
- `landsatNDVI_precip.js`:
- `sentinelCBackscatter.js`:
- `palsarLBackscatter.js`:
- `srtmTopography.js`:
  
**`main/`**: Python and R scripts used for main processing and computing. 

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


