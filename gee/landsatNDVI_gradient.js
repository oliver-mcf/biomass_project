
// GEE Script to Retrieve NDVI Gradient

/////////////////// LANDSAT NDVI GRADIENT /////////////////// 
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
  
// Remove clouds and shadows
function maskClouds(image) {
  var qa = image.select('QA_PIXEL');
  // Bit 3 (cloud), Bit 5 (shadow)
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

// Calcuate monthly median NDVI and label months
var calculateMonthlyNDVI = function(monthOffset) {
  var startDate = start.advance(monthOffset, 'month');
  var endDate = startDate.advance(1, 'month');
  var monthNumber = ee.Number(monthOffset).add(1);
  var medianNDVI = landsatNDVI.select('ndvi').filterDate(startDate, endDate).median().rename('ndvi');
  var monthBand = ee.Image(monthNumber).toInt().rename('month');
  return medianNDVI.addBands(monthBand).set('system:time_start', startDate.millis())};
var monthlyNDVIList = ee.List.sequence(0, 11).map(calculateMonthlyNDVI);
var landsatMonthNDVI = ee.ImageCollection.fromImages(monthlyNDVIList);
print(landsatMonthNDVI);

// Filter no data
var landsatGradient = landsatMonthNDVI.filter(ee.Filter.listContains('system:band_names', 'ndvi'));

// Calculate percentiles
var MaxMin = landsatGradient.select('ndvi').reduce(ee.Reducer.percentile([5,95]));

// Attach percentiles to months
landsatGradient = landsatGradient.map(function(img){
  var min_ = img.select('ndvi').eq(MaxMin.select('ndvi_p5')).multiply(img.select('month'));
  var max_ = img.select('ndvi').eq(MaxMin.select('ndvi_p95')).multiply(img.select('month'));
  return img.addBands(min_.rename('p5month')).addBands(max_.rename('p95month'))});
print(landsatGradient);

// Isolate percentiles
var MinImage = landsatGradient.select('p5month').reduce(ee.Reducer.max());
var MaxImage = landsatGradient.select('p95month').reduce(ee.Reducer.max());

// Calculate gradient
var diff = MaxImage.subtract(MinImage).abs();
var change = MaxMin.select('ndvi_p95').subtract(MaxMin.select('ndvi_p5'));
var gradient = change.divide(diff).rename('gradient');
Map.addLayer(gradient.clip(roi));
Export.image.toDrive({image: gradient, description: '19_NDVI_Gradient', region: roi, scale: 30, 
                      folder: 'TKW'});


