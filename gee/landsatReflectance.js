
// GEE Script to Retrieve Landsat Band Reflectances

/////////////////// LANDSAT BAND REFLECTANCE /////////////////// 
// Define temporal and spatial bounds
var roi = MGR;
Map.addLayer(roi);
Map.centerObject(roi);
var landsat = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
  .filterMetadata('CLOUD_COVER', 'less_than', 30)
  .filterDate('2018-08-01', '2019-07-31')
  .filterBounds(roi);
  
// Define a function to mask clouds and shadows
function maskClouds(image) {
  var qa = image.select('QA_PIXEL');
  // Bits 3 and 5 are cloud and shadow respectively
  var mask = qa.bitwiseAnd(1 << 3).eq(0).and(qa.bitwiseAnd(1 << 5).eq(0));
  return image.updateMask(mask);
}

// Apply cloud masking function to the collection
var landsatMasked = landsat.map(maskClouds);

// Define a function to calculate reflectance statistics and export images
function exportBandStatistics(year) {
  // Isolate bands
  var bandNames = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'];
  // Loop over bands and calculate statistics
  bandNames.forEach(function(band) {
    var bandImg = landsatMasked.select(band);
    var median = bandImg.median();
    var stDev = bandImg.reduce(ee.Reducer.stdDev());
    Map.addLayer(stDev.clip(roi));
    var p95 = bandImg.reduce(ee.Reducer.percentile([95]));
    var p05 = bandImg.reduce(ee.Reducer.percentile([5]));
    // Export 
    Export.image.toDrive({image: median.toFloat(), description: year + '_' + band + '_Median', region: roi, scale: 30, folder: 'MGR'});
    Export.image.toDrive({image: stDev.toFloat(), description: year + '_' + band + '_StDev', region: roi, scale: 30, folder: 'MGR'});
    Export.image.toDrive({image: p95.toFloat(), description: year + '_' + band + '_p95', region: roi, scale: 30, folder: 'MGR'});
    Export.image.toDrive({image: p05.toFloat(), description: year + '_' + band + '_p05', region: roi, scale: 30, folder: 'MGR'});
  });
}

// Call the function for a given year
exportBandStatistics('19');