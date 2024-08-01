
// GEE Script to Retrieve L-Band Backscatter from PALSAR-2 ScanSAR

/////////////////// PALSAR BACKSCATTER /////////////////// 
var roi = MGR;
Map.addLayer(roi);
Map.centerObject(roi);

// Define temporal and spatial bounds
var start = ee.Date('2018-08-01');
var end = ee.Date('2019-07-31');
var alos = ee.ImageCollection("JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR")
  .filterDate(start, end)
  .filterBounds(roi);

// Keep images with both HH and HV polarisations
var hasBands = function(image) {
  var bandNames = image.bandNames();
  return bandNames.contains('HH').and(bandNames.contains('HV'));
};
var palsar = alos.filter(ee.Filter.and(
  ee.Filter.listContains('system:band_names', 'HH'),
  ee.Filter.listContains('system:band_names', 'HV')));
//print(palsar);

// Calculate Ratio and Index
var calculateRatioAndIndex = function(image) {
  var hh = image.select('HH');
  var hv = image.select('HV');
  var ratio = hh.divide(hv).rename('Ratio');
  var index = image.normalizedDifference(['HH', 'HV']).rename('Index');
  return image.addBands(ratio).addBands(index);
};
var palsarWithRatioAndIndex = palsar.map(calculateRatioAndIndex);

// Derive statistics from band in collection
function computeStats(collection, band) {
  var median = collection.select(band).median().rename(band + '_Median').clip(roi);
  var stdDev = collection.select(band).reduce(ee.Reducer.stdDev()).rename(band + '_StDev').clip(roi);
  var p95 = collection.select(band).reduce(ee.Reducer.percentile([95])).rename(band + '_95').clip(roi);
  var p05 = collection.select(band).reduce(ee.Reducer.percentile([5])).rename(band + '_05').clip(roi);
  return ee.Image.cat([median, stdDev, p95, p05])}
var HV_Stats = computeStats(palsar, 'HV');
var HH_Stats = computeStats(palsar, 'HH');
Map.addLayer(HV_Stats);
Map.addLayer(HH_Stats);

// Export each metric as an image
function exportImage(image, band, year, region, folder) {
  Export.image.toDrive({image: image.select(band + '_Median'), description: year + '_' + band + '_Median', region: region, scale: 25, maxPixels: 1e10, folder: folder});
  Export.image.toDrive({image: image.select(band + '_StDev'), description: year + '_' + band + '_StDev', region: region, scale: 25, maxPixels: 1e10, folder: folder});
  Export.image.toDrive({image: image.select(band + '_95'), description: year + '_' + band + '_95', region: region, scale: 25, maxPixels: 1e10, folder: folder});
  Export.image.toDrive({image: image.select(band + '_05'), description: year + '_' + band + '_05', region: region, scale: 25, maxPixels: 1e10, folder: folder})}
exportImage(HV_Stats, 'HV', '19', roi, 'MGR');
exportImage(HH_Stats, 'HH', '19', roi, 'MGR');

// Select the ratio and index images
var ratioImage = palsarWithRatioAndIndex.select('Ratio').median().clip(roi);
var indexImage = palsarWithRatioAndIndex.select('Index').median().clip(roi);
Map.addLayer(ratioImage, {min: 0, max: 1}, 'Ratio');
Map.addLayer(indexImage, {min: -1, max: 1}, 'Index');
Export.image.toDrive({image: ratioImage, description: '19_HHHV_Ratio', region: roi, scale: 30, 
                      folder: 'MGR'});
Export.image.toDrive({image: indexImage, description: '19_HHHV_Index', region: roi, scale: 30, 
                      folder: 'MGR'});

