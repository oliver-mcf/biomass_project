
// GEE Script to Retrieve NDVI Median

/////////////////// LANDSAT NDVI MEDIAN /////////////////// 
// Define temporal and spatial bounds
var roi = TKW;
Map.addLayer(roi);
Map.centerObject(roi);
var start = ee.Date('2018-08-01');
var end = ee.Date('2019-07-31');

// Load landsat data
var LANDSAT = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2");
var landsat = LANDSAT.filterMetadata('CLOUD_COVER', 'less_than', 50).filterDate(start, end).filterBounds(roi);
  
// Mask clouds and shadows
function maskClouds(image) {
  var qa = image.select('QA_PIXEL');
  // Bits 3 and 5 are cloud and shadow respectively
  var mask = qa.bitwiseAnd(1 << 3).eq(0).and(qa.bitwiseAnd(1 << 5).eq(0));
  return image.updateMask(mask)}
var landsatMasked = landsat.map(maskClouds);

// Add NDVI band to each Landsat image
var addNDVI = function(image) {
  var ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('ndvi');
  return image.addBands(ndvi)};
var landsatNDVI = landsatMasked.map(addNDVI);

// Calculate median NDVI for a specific term period
var calculateMedianNDVI = function(startDate, endDate) {
  return landsatNDVI.filterDate(startDate, endDate).select('ndvi').median().rename('ndvi_median')};
var period = [{ name: 'T1', startDate: '2018-08-01', endDate: '2018-11-01' },
              { name: 'T2', startDate: '2018-12-01', endDate: '2019-03-01' },
              { name: 'T3', startDate: '2019-04-01', endDate: '2019-07-01' },
              { name: 'YR', startDate: start, endDate: end}];

// Export median NDVI images
period.forEach(function(image) {
  var medianNDVI = calculateMedianNDVI(image.startDate, image.endDate);
  Map.addLayer(medianNDVI.clip(roi));
  Export.image.toDrive({image: medianNDVI, description: '19_' + image.name + '_NDVI_Median', 
                        region: roi, crs: 'EPSG:3857', scale: 30, folder: 'TKW' })});
