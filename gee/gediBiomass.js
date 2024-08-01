
// GEE Script to Retrieve GEDI L4A AGB

/////////////////// GEDI HEIGHT, COVER, BIOMASS /////////////////// 
// Define temporal and spatial bounds
var roi = MGR; // or TKW
Map.centerObject(roi);
Map.addLayer(roi);
var year = '23';
var site = 'MGR';
var start = ee.Date('2022-11-01');
var end = ee.Date('2023-04-30');

// Read EO datasets
var GEDI = ee.ImageCollection("LARSE/GEDI/GEDI04_A_002_MONTHLY");
var HEIGHT = ee.ImageCollection("LARSE/GEDI/GEDI02_A_002_MONTHLY");
var COVER = ee.ImageCollection("LARSE/GEDI/GEDI02_B_002_MONTHLY");

// Load the GEDI, height, and cover image collections
var gedi = GEDI.filterDate(start, end).filterBounds(roi);
print("gedi", gedi);
var height = HEIGHT.filterDate(start, end).filterBounds(roi);
print("height", height);
var cover = COVER.filterDate(start, end).filterBounds(roi);
print("cover", cover);

// Define a function to join images by system:index and combine bands
var combineBands = function(gediImage) {
  var index = gediImage.get('system:index');
  var heightImage = height.filter(ee.Filter.equals('system:index', index)).first();
  var coverImage = cover.filter(ee.Filter.equals('system:index', index)).first();
  heightImage = heightImage ? heightImage.select('rh98') : ee.Image.constant(0).rename('rh98');
  coverImage = coverImage ? coverImage.select('cover') : ee.Image.constant(0).rename('cover');
  var gediBands = gediImage.select(['agbd', 'leaf_off_flag', 'degrade_flag', 'algorithm_run_flag', 'l4_quality_flag', 'l2_quality_flag', 'sensitivity']);
  var combinedImage = gediBands.addBands(heightImage).addBands(coverImage);
  return combinedImage.set('system:time_start', gediImage.get('system:time_start')).set('system:index', index);
};

// Map the combineBands function over the GEDI image collection
var combinedCollection = gedi.map(combineBands);
print("combinedCollection", combinedCollection);

// Function to count pixel values
var countPixels = function(img) {
  var maskedImg = img.updateMask(img.select('agbd').gt(0));
  var pixelCount = maskedImg.reduceRegion({
    reducer: ee.Reducer.count(),
    geometry: roi,
    scale: 25,
    crs: 'EPSG:3857',
    maxPixels: 1e10
  }).get('agbd');
  return img.set('pixel_count', pixelCount);
};

// Initial footprints
var allPixels = combinedCollection.map(countPixels);
var allPixelsTotal = allPixels.aggregate_sum('pixel_count');
print('INITIAL:', allPixelsTotal);

// Function to add month property to each image (for mean)
var addMonth = function(img) {
  var date = ee.Date(img.get('system:time_start'));
  var month = date.format('YYYY-MM');
  return img.set('month', month);
};
combinedCollection = combinedCollection.map(addMonth);

// Function to quality filter
var qualityFilter = function(img) {
  var mask = img.select('rh98').gt(3) // Li et al (2023)
    .and(img.select('agbd').lte(300)) // biologically implausible
    .and(img.select('sensitivity').gte(0.98)) // Li et al (2023)
    .and(img.select('l2_quality_flag').eq(1))
    .and(img.select('l4_quality_flag').eq(1))
    .and(img.select('degrade_flag').eq(0))
    .and(img.select('algorithm_run_flag').eq(1));
  return img.updateMask(mask);
};

var gediFiltered = combinedCollection.map(qualityFilter).map(addMonth);
print('gediFiltered', gediFiltered);

// Filtered footprints
var allPixelsFiltered = gediFiltered.map(countPixels);
var allPixelsTotalFiltered = allPixelsFiltered.aggregate_sum('pixel_count');
print('FILTERED:', allPixelsTotalFiltered);

// Export annual composite of filtered GEDI L4A Biomass
var gediAGB = gediFiltered.select(['agbd']);
var mosaicAGB = gediAGB.mosaic().clip(roi);
Map.addLayer(mosaicAGB, {min: 0, max: 300, palette: ['white', 'blue']}, 'Mosaic AGB Filtered');
Export.image.toDrive({
  image: mosaicAGB, 
  description: year + '_GEDI_AGB', 
  region: roi, 
  crs: 'EPSG:3857',
  scale: 25, 
  folder: site,
});

// Export annual composite of GEDI L2B Cover (filtered footprints)
var gediCover = gediFiltered.select(['cover']);
var mosaicCover = gediCover.mosaic().clip(roi);
Map.addLayer(mosaicCover, {min: 0, max: 1, palette: ['white', 'green']}, 'Mosaic Cover Filtered');
Export.image.toDrive({
  image: mosaicCover, 
  description: year + '_GEDI_COVER', 
  region: roi,
  crs: 'EPSG:3857',
  scale: 25, 
  folder: site,
});

// Function to compute mean AGBD value
var computeMeanAGBD = function(monthlyCollection) {
  var mean = monthlyCollection.mean().select('agbd');
  return mean.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: roi,
    scale: 25,
    crs: 'EPSG:3857',
    maxPixels: 1e10
  }).get('agbd');
};

// Function to group by month and compute mean
var getMonthlyMeans = function(collection) {
  var months = ee.List(collection.aggregate_array('month')).distinct();
  var monthlyMeans = months.map(function(month) {
    var monthlyCollection = collection.filter(ee.Filter.eq('month', month));
    var meanAGBD = computeMeanAGBD(monthlyCollection);
    return ee.Feature(null, {'month': month, 'mean_agbd': meanAGBD});
  });
  return ee.FeatureCollection(monthlyMeans);
};

// Compute monthly mean of initial data
var combinedMonthlyMeans = getMonthlyMeans(combinedCollection);
print("INITIAL MEANS:", combinedMonthlyMeans);
Export.table.toDrive({
  collection: combinedMonthlyMeans,
  description: 'initial_monthly_mean',
  fileFormat: 'CSV'
});

// Compute monthly mean of filtered data
var filteredMonthlyMeans = getMonthlyMeans(gediFiltered);
print("FILTERED MEANS:", filteredMonthlyMeans);
Export.table.toDrive({
  collection: filteredMonthlyMeans,
  description: 'filtered_monthly_mean',
  fileFormat: 'CSV'
});


