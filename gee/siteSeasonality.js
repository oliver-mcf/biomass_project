
// GEE Script to Retrieve Summary of Site Seasonality


/////////////////// ANNUAL PRECIPITATION /////////////////// 
// Define params
var roi = TKW; // or MGR
var start = ee.Date('2016-08-01');
var end = ee.Date('2023-07-31');

// Import precipitation data
var era = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
  .filterDate(start, end)
  .filterBounds(roi);
var eraPrcp = era.map(function(img) {
  return img.select('total_precipitation_sum').multiply(1000).rename('precip').set('system:time_start', img.get('system:time_start'))});

// Filter the dataset for the specified date range
var filteredData = eraPrcp.filterDate(start, end);

var meanImage = filteredData.sum().rename('total_precip');

// Reduce the image to a single value by taking the mean over the ROI
var stats = meanImage.reduceRegion({
  reducer: ee.Reducer.mean(),
  geometry: roi,
  maxPixels: 1e10
});

// Print the mean annual precipitation
print('Average Annual Precipitation (mm):', stats.get('total_precip'));


/////////////////// MONTHLY PRECIPITATION /////////////////// 
// Calculate the monthly mean precipitation for the ROI
var monthlyMeanPrcp = eraPrcp.map(function(img) {
  var date = img.date().format('YYYY-MM');
  var meanPrcp = img.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: roi,
    maxPixels: 1e10
  }).get('precip');
  
  return ee.Feature(null, {
    'date': date,
    'mean_precip_mm': meanPrcp
  });
});

// Print the monthly mean precipitation values to the console
monthlyMeanPrcp.evaluate(function(result) {
  print('Monthly Mean Precipitation:', result.features.map(function(f) {
    return {
      date: f.properties.date,
      mean_precip_mm: f.properties.mean_precip_mm
    };
  }));
});

// Convert the FeatureCollection to a CSV
Export.table.toDrive({
  collection: ee.FeatureCollection(monthlyMeanPrcp),
  description: 'AverageMonthlyPrecipitation',
  folder: 'TKW',
  fileFormat: 'CSV',
  selectors: ['date', 'mean_precip_mm']
});

// Print the FeatureCollection to the console
print(monthlyMeanPrcp);


/////////////////// MONTHLY NDVI /////////////////// 
var landsat = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
  .filterMetadata('CLOUD_COVER', 'less_than', 30)
  .filterDate(startDate, endDate)
  .filterBounds(roi);

// Remove clouds (3) and shadows (5)
function maskClouds(image) {
  var qa = image.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 3).eq(0).and(qa.bitwiseAnd(1 << 5).eq(0));
  return image.updateMask(mask);
}
var landsatMasked = landsat.map(maskClouds);

// Add NDVI band to images
function addNDVI(image) {
  var ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI');
  return image.addBands(ndvi);
}
var landsatNDVI = landsatMasked.map(addNDVI);


// Calculate median monthly NDVI
var months = ee.List.sequence(0, 11);
var monthNames = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'];

var getMonthlyNDVI = function(monthOffset) {
  var start = ee.Date(startDate).advance(monthOffset, 'month');
  var end = start.advance(1, 'month');
  var monthNDVI = landsatNDVI.filterDate(start, end).select('NDVI').median().rename('NDVI');
  return monthNDVI.set('month', monthNames[monthOffset]);
};

var monthlyNDVIs = ee.ImageCollection.fromImages(
  months.map(getMonthlyNDVI)
);

// Convert the ImageCollection to a FeatureCollection with NDVI values
var monthlyNDVIFeatures = monthlyNDVIs.map(function(img) {
  var ndviValue = img.reduceRegion({
    reducer: ee.Reducer.median(),
    geometry: roi,
    scale: 30,
    maxPixels: 1e10
  }).get('NDVI');
  
  // Add error handling if NDVI value is missing
  return ee.Feature(null, {
    month: img.get('month'),
    NDVI: ndviValue ? ndviValue : 'No NDVI Value'
  });
});


// Export the results to a CSV file
Export.table.toDrive({
  collection: monthlyNDVIFeatures,
  description: 'MonthNDVI',
  folder: 'MGR',
  fileFormat: 'CSV',
  selectors: ['month', 'NDVI']
});