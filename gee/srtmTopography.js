
// GEE Script to Retrieve Topographic Variables

/////////////////// SRTM TOPOGRAPHY /////////////////// 
// Define params
srtm = ee.Image("CGIAR/SRTM90_V4");
mtpi = ee.Image("CSP/ERGo/1_0/Global/SRTM_mTPI");
Map.centerObject(roi);
Map.addLayer(roi);

// Derive topography metrics
var elevation = srtm.select('elevation').clip(roi);
var slope = ee.Terrain.slope(elevation).clip(roi);
var tpi = mtpi.select('elevation').clip(roi);

// Export topography metrics
Export.image.toDrive({image: elevation, description: 'SRTM_Elevation', region: roi, scale: 90, folder: 'MGR' });
Export.image.toDrive({image: slope, description: 'SRTM_Slope', region: roi, scale: 90, folder: 'MGR' });
Export.image.toDrive({image: tpi, description: 'SRTM_mTPI', region: roi, scale: 270, folder: 'MGR' });