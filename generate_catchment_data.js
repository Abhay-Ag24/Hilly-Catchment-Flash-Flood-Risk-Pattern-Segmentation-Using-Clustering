/**
 * Data Generation Script: IMD Rainfall & NASA/USGS SRTM Catchment Data
 * Generates realistic geomorphological and meteorological records for 1,200
 * hilly catchments in the Indian Himalayas and Western Ghats.
 */

const fs = require('fs');
const path = require('path');

function randNormal(mean, stdDev) {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  const num = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  return num * stdDev + mean;
}

function clamp(val, min, max) {
  return Math.max(min, Math.min(max, val));
}

const REGIONS = [
  { name: 'Upper_Ganga_Himalaya', weight: 0.30, baseElev: 2200, elevStd: 600, baseSlope: 28, baseRain: 180, baseInt: 45 },
  { name: 'Western_Ghats_Windward', weight: 0.25, baseElev: 1100, elevStd: 350, baseSlope: 24, baseRain: 290, baseInt: 55 },
  { name: 'Teesta_Eastern_Himalaya', weight: 0.20, baseElev: 1850, elevStd: 500, baseSlope: 26, baseRain: 240, baseInt: 50 },
  { name: 'Beas_Western_Himalaya', weight: 0.15, baseElev: 1950, elevStd: 450, baseSlope: 25, baseRain: 160, baseInt: 40 },
  { name: 'Western_Ghats_Leeward', weight: 0.10, baseElev: 850, elevStd: 250, baseSlope: 14, baseRain: 95, baseInt: 22 }
];

const TERRAIN_ZONES = [
  'High_Himalaya_Glaciated',
  'Lesser_Himalaya_Escarpment',
  'Western_Ghats_Windward_Ridge',
  'Western_Ghats_Leeward_Plateau',
  'Siwalik_Foothills'
];

const SOIL_TYPES = [
  'Excessively_Drained_Gravel',
  'Moderately_Well_Drained_Loam',
  'Poorly_Drained_Clayey_Shale'
];

const VEGETATION_TYPES = [
  'Dense_Canopy',
  'Degraded_Forest_Scrub',
  'Agricultural_Terrace',
  'Barren_Exposed_Rock'
];

function generateCatchments(count = 1200) {
  const records = [];

  for (let i = 1; i <= count; i++) {
    const id = `CATCH_${String(i).padStart(4, '0')}`;

    // Select Region based on weights
    const rRoll = Math.random();
    let accumulated = 0;
    let region = REGIONS[0];
    for (const r of REGIONS) {
      accumulated += r.weight;
      if (rRoll <= accumulated) {
        region = r;
        break;
      }
    }

    // Geomorphological SRTM attributes
    const elevation_mean_m = Math.round(clamp(randNormal(region.baseElev, region.elevStd), 400, 3900));
    const reliefRatio = clamp(randNormal(0.48, 0.12), 0.2, 0.85);
    const elevation_relief_m = Math.round(elevation_mean_m * reliefRatio);
    const slope_mean_deg = parseFloat(clamp(randNormal(region.baseSlope, 6.5), 6.0, 48.0).toFixed(2));
    const drainage_density_km_sqkm = parseFloat(clamp(randNormal(2.8 + (slope_mean_deg / 15), 0.65), 1.1, 5.8).toFixed(2));
    
    // TWI = ln(a / tan(beta)). Higher on gentle slopes, lower on steep slopes
    const slopeRad = (slope_mean_deg * Math.PI) / 180;
    const twiBase = Math.log(120 / Math.max(0.08, Math.tan(slopeRad)));
    const topographic_wetness_index = parseFloat(clamp(randNormal(twiBase, 1.1), 3.2, 14.8).toFixed(2));

    const hypsometric_integral = parseFloat(clamp(randNormal(0.52 + (elevation_mean_m > 2000 ? 0.1 : -0.05), 0.11), 0.22, 0.84).toFixed(3));
    const ruggedness_number = parseFloat((elevation_relief_m * drainage_density_km_sqkm / 1000).toFixed(2));
    const catchment_area_sqkm = parseFloat(clamp(randNormal(115, 65), 12.5, 480.0).toFixed(1));

    // IMD Meteorological attributes
    const peak_monsoon_rainfall_mm = parseFloat(clamp(randNormal(region.baseRain + (slope_mean_deg * 2.2), 48), 65, 480).toFixed(1));
    const max_hourly_intensity_mm_hr = parseFloat(clamp(randNormal(region.baseInt + (peak_monsoon_rainfall_mm / 10), 12.5), 14.0, 96.0).toFixed(1));
    const consecutive_wet_days_cwd = Math.round(clamp(randNormal(region.name.includes('Windward') ? 14 : 9, 3.8), 2, 28));
    const precipitation_concentration_index = parseFloat(clamp(randNormal(18.5, 4.2), 11.0, 32.5).toFixed(2));

    // Kirpich's formula for response time / time of concentration
    // Tc ~ 0.0195 * (L^0.77) * (S^-0.385)
    const streamLenKm = Math.sqrt(catchment_area_sqkm) * 1.35;
    const slopeSlope = Math.tan(slopeRad);
    const tcApprox = (0.06 * Math.pow(streamLenKm * 1000, 0.77) / Math.pow(slopeSlope, 0.385)) / 60;
    const response_time_hr = parseFloat(clamp(randNormal(tcApprox, tcApprox * 0.2), 0.8, 14.5).toFixed(2));

    // Categorical variables
    let terrain_zone = 'Siwalik_Foothills';
    if (region.name.includes('Ghats')) {
      terrain_zone = region.name.includes('Windward') ? 'Western_Ghats_Windward_Ridge' : 'Western_Ghats_Leeward_Plateau';
    } else {
      if (elevation_mean_m > 2600) terrain_zone = 'High_Himalaya_Glaciated';
      else if (elevation_mean_m > 1400) terrain_zone = 'Lesser_Himalaya_Escarpment';
      else terrain_zone = 'Siwalik_Foothills';
    }

    const soilRoll = Math.random();
    const soil_drainage_type = slope_mean_deg > 28 ? (soilRoll < 0.65 ? SOIL_TYPES[0] : SOIL_TYPES[1]) : (soilRoll < 0.5 ? SOIL_TYPES[1] : SOIL_TYPES[2]);

    const vegRoll = Math.random();
    let vegetation_cover_class = VEGETATION_TYPES[0];
    if (elevation_mean_m > 2800 || slope_mean_deg > 35) {
      vegetation_cover_class = vegRoll < 0.6 ? VEGETATION_TYPES[3] : VEGETATION_TYPES[1];
    } else if (region.name.includes('Windward')) {
      vegetation_cover_class = vegRoll < 0.7 ? VEGETATION_TYPES[0] : VEGETATION_TYPES[1];
    } else {
      vegetation_cover_class = vegRoll < 0.4 ? VEGETATION_TYPES[2] : (vegRoll < 0.75 ? VEGETATION_TYPES[1] : VEGETATION_TYPES[0]);
    }

    // Realistic missing values (~3.5% rate) for genuine Objective 1 preprocessing
    const pRainVal = Math.random() < 0.035 ? '' : peak_monsoon_rainfall_mm;
    const pIntVal = Math.random() < 0.035 ? '' : max_hourly_intensity_mm_hr;
    const twiVal = Math.random() < 0.03 ? '' : topographic_wetness_index;
    const soilVal = Math.random() < 0.025 ? '' : soil_drainage_type;

    records.push({
      catchment_id: id,
      basin_region: region.name,
      terrain_zone,
      soil_drainage_type: soilVal,
      vegetation_cover_class,
      elevation_mean_m,
      elevation_relief_m,
      slope_mean_deg,
      drainage_density_km_sqkm,
      topographic_wetness_index: twiVal,
      hypsometric_integral,
      ruggedness_number,
      catchment_area_sqkm,
      peak_monsoon_rainfall_mm: pRainVal,
      max_hourly_intensity_mm_hr: pIntVal,
      consecutive_wet_days_cwd,
      precipitation_concentration_index,
      response_time_hr
    });
  }

  return records;
}

function recordsToCSV(records) {
  const headers = Object.keys(records[0]);
  const rows = records.map(rec => headers.map(h => rec[h] !== null && rec[h] !== undefined ? rec[h] : '').join(','));
  return [headers.join(','), ...rows].join('\n');
}

const dataDir = path.join(__dirname, '..', 'data');
if (!fs.existsSync(dataDir)) {
  fs.mkdirSync(dataDir, { recursive: true });
}

const records = generateCatchments(1200);
const csvContent = recordsToCSV(records);
const outputPath = path.join(dataDir, 'raw_imd_srtm_catchment_data.csv');

fs.writeFileSync(outputPath, csvContent, 'utf8');
console.log(`Generated ${records.length} raw catchment records successfully at: ${outputPath}`);
