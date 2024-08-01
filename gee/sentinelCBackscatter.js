
// GEE Script to Retrieve Sentinel-1 C-Band Backscatter

/////////////////// SENTINEL-1 BACKSCATTER /////////////////// 
var roi = MGR;
Map.addLayer(roi);
Map.centerObject(roi);

// Define temporal and spatial bounds
var start = ee.Date('2015-08-01');
var end = ee.Date('2016-07-31');
var sent = ee.ImageCollection("COPERNICUS/S1_GRD")
  .filterDate(start, end)
  .filterBounds(roi)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'));
print(sent);

// Derive statistics from band in collection
function computeStats(collection, band) {
  var median = collection.select(band).median().rename(band + '_Median').clip(roi);
  var stdDev = collection.select(band).reduce(ee.Reducer.stdDev()).rename(band + '_StDev').clip(roi);
  var p95 = collection.select(band).reduce(ee.Reducer.percentile([95])).rename(band + '_95').clip(roi);
  var p05 = collection.select(band).reduce(ee.Reducer.percentile([5])).rename(band + '_05').clip(roi);
  return ee.Image.cat([median, stdDev, p95, p05])}
var VV_Stats = computeStats(sent, 'VV');
var VH_Stats = computeStats(sent, 'VH');
Map.addLayer(VV_Stats);
Map.addLayer(VH_Stats);

// Export each metric as an image
function exportImage(image, band, year, region, folder) {
  Export.image.toDrive({image: image.select(band + '_Median'), description: year + '_' + band + '_Median', region: region, scale: 25, maxPixels: 1e10, folder: folder});
  Export.image.toDrive({image: image.select(band + '_StDev'), description: year + '_' + band + '_StDev', region: region, scale: 25, maxPixels: 1e10, folder: folder});
  Export.image.toDrive({image: image.select(band + '_95'), description: year + '_' + band + '_95', region: region, scale: 25, maxPixels: 1e10, folder: folder});
  Export.image.toDrive({image: image.select(band + '_05'), description: year + '_' + band + '_05', region: region, scale: 25, maxPixels: 1e10, folder: folder});
}
exportImage(VV_Stats, 'VV', '16', roi, 'MGR');
exportImage(VH_Stats, 'VH', '16', roi, 'MGR');
