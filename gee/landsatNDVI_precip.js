
// GEE Script to Retrieve NDVI Percentiles with Precipitation Conditions

/////////////////// LANDSAT NDVI PRECIPITATION /////////////////// 
var roi = TKW;
Map.addLayer(roi);
Map.centerObject(roi);

// Define temporal and spatial bounds
var start = ee.Date('2018-08-01');
var end = ee.Date('2019-07-31');
var landsat = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
  .filterMetadata('CLOUD_COVER', 'less_than', 30)
  .filterDate(start, end)
  .filterBounds(roi);
  
// Remove clouds (3) and shadows (5)
function maskClouds(image) {
  var qa = image.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 3).eq(0).and(qa.bitwiseAnd(1 << 5).eq(0));
  return image.updateMask(mask)}
var landsatMasked = landsat.map(maskClouds);

// Append NDVI
var addNDVI = function(image) {
  var ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('ndvi');
  return image.addBands(ndvi)};
var landsatNDVI = landsatMasked.map(addNDVI);
print(landsatNDVI);

// Visualise NDVI
var timeNDVI = ui.Chart.image.seriesByRegion(landsatNDVI.select('ndvi'),roi,ee.Reducer.median(),'ndvi',30);
print(timeNDVI);

// Calculate monthly median NDVI
var calculateMonthlyNDVI = function(monthOffset) {
  var startDate = start.advance(monthOffset, 'month');
  var endDate = startDate.advance(1, 'month');
  var medianNDVI = landsatNDVI.select('ndvi').filterDate(startDate, endDate).median().rename('ndvi');
  return medianNDVI.set('month', startDate.format('YYYYMM'))};
var monthlyNDVIList = ee.List.sequence(0, 11).map(calculateMonthlyNDVI);
var landsatMonthNDVI_initial = ee.ImageCollection.fromImages(monthlyNDVIList);
print("landsatMonthNDVI_initial", landsatMonthNDVI_initial);

// Filter no data months
var landsatMonthNDVI = landsatMonthNDVI_initial.filter(ee.Filter.listContains('system:band_names', 'ndvi'));
print("landsatMonthNDVI", landsatMonthNDVI);

// Import precipitation data
var era = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
  .filterDate(start, end)
  .filterBounds(roi);
var eraPrcp = era.map(function(img) {
  return img.select('total_precipitation_sum').multiply(1000).rename('precip').set('system:time_start', img.get('system:time_start'))});

// Visualise precipitation data
var timeRain = ui.Chart.image.seriesByRegion(eraPrcp.select('precip'), roi, ee.Reducer.median(), 'precip');
print(timeRain);

// Identify wet/dry months
var eraPrcpWetMask = eraPrcp.map(function(img) {
  var wet = img.gte(30).rename('wetmonth');
  var dry = img.lt(30).rename('drymonth');
  return wet.where(wet, 1).addBands(dry.where(dry, 1))});
print("eraPrcpWetMask", eraPrcpWetMask);

// Align datasets by month
var validMonths = landsatMonthNDVI.aggregate_array('month');
var eraMonthPrcp = eraPrcpWetMask.filter(ee.Filter.inList('system:index', validMonths));
print("eraMonthWet:", eraMonthPrcp);
print("landsatMonthNDVI", landsatMonthNDVI);

// Append precipitation mask to NDVI collection
var appendBands = function(image) {
  var alignedImage = eraMonthPrcp.filterMetadata('system:index', 'equals', image.get('month')).first();
  var result = image.addBands(alignedImage);
  return result};
var landsatPrecip = landsatMonthNDVI.map(appendBands);
print("landsatPrecip", landsatPrecip);

// Isolate wet NDVI pixels 
var wetMask = function(image) {
  var wetmonth = image.select('wetmonth');
  var mask = wetmonth.eq(1);
  var maskedNDVI = image.select('ndvi').updateMask(mask);
  return maskedNDVI};
var NDVI_WET = landsatPrecip.map(wetMask);
print("NDVI_WET:", NDVI_WET);

// Calculate wet percentiles
var ndvi_wet95 = NDVI_WET.select('ndvi').reduce(ee.Reducer.percentile([95]));
var ndvi_wet05 = NDVI_WET.select('ndvi').reduce(ee.Reducer.percentile([5]));
var pallette = {min: -1, max: 1, palette: ['blue', 'white', 'green']};
Map.addLayer(ndvi_wet95, pallette, "WET_95");
Map.addLayer(ndvi_wet05, pallette, "WET_5");

// Isolate dry NDVI pixels 
var dryMask = function(image) {
  var drymonth = image.select('drymonth');
  var mask = drymonth.eq(1);
  var maskedNDVI = image.select('ndvi').updateMask(mask);
  return maskedNDVI};
var NDVI_DRY = landsatPrecip.map(dryMask);
print("NDVI_DRY:", NDVI_DRY);

// Calculate dry percentiles
var ndvi_dry95 = NDVI_DRY.select('ndvi').reduce(ee.Reducer.percentile([95]));
var ndvi_dry05 = NDVI_DRY.select('ndvi').reduce(ee.Reducer.percentile([5]));
var pallette = {min: -1, max: 1, palette: ['blue', 'white', 'green']};
Map.addLayer(ndvi_dry95, pallette, "DRY_95");
Map.addLayer(ndvi_dry05, pallette, "DRY_5");

// Export percentiles
Export.image.toDrive({image: ndvi_wet95, description: '19_NDVI_Wet95', 
                      region: roi, scale: 30, folder: 'TKW'});
Export.image.toDrive({image: ndvi_wet95, description: '19_NDVI_Wet05', 
                      region: roi, scale: 30, folder: 'TKW'});
Export.image.toDrive({image: ndvi_dry95, description: '19_NDVI_Dry95', 
                      region: roi, scale: 30, folder: 'TKW'});
Export.image.toDrive({image: ndvi_dry95, description: '19_NDVI_Dry05', 
                      region: roi, scale: 30, folder: 'TKW'});