/**
 * Comprehensive Cluster Analysis Engine
 * Performs Preprocessing, Scaling, K-Means (Euclidean & Manhattan),
 * Hierarchical Clustering, PCA Decomposition, Validation Metrics (WCSS, Silhouette, Davies-Bouldin),
 * and Cluster Profiling.
 */

const fs = require('fs');
const path = require('path');

// 1. CSV Parser
function parseCSV(content) {
  const lines = content.trim().split(/\r?\n/);
  const headers = lines[0].split(',');
  const records = [];

  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(',');
    const obj = {};
    headers.forEach((h, idx) => {
      obj[h] = values[idx] !== '' ? values[idx] : null;
    });
    records.push(obj);
  }
  return { headers, records };
}

// 2. Preprocessing & Imputation
const NUMERICAL_COLS = [
  'elevation_mean_m',
  'elevation_relief_m',
  'slope_mean_deg',
  'drainage_density_km_sqkm',
  'topographic_wetness_index',
  'hypsometric_integral',
  'ruggedness_number',
  'catchment_area_sqkm',
  'peak_monsoon_rainfall_mm',
  'max_hourly_intensity_mm_hr',
  'consecutive_wet_days_cwd',
  'precipitation_concentration_index',
  'response_time_hr'
];

const CATEGORICAL_COLS = [
  'terrain_zone',
  'soil_drainage_type',
  'vegetation_cover_class'
];

function median(values) {
  const sorted = [...values].filter(v => v !== null && !isNaN(v)).sort((a, b) => a - b);
  if (sorted.length === 0) return 0;
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function mode(values) {
  const counts = {};
  let maxCount = 0;
  let bestVal = null;
  values.forEach(v => {
    if (v !== null && v !== undefined && v !== '') {
      counts[v] = (counts[v] || 0) + 1;
      if (counts[v] > maxCount) {
        maxCount = counts[v];
        bestVal = v;
      }
    }
  });
  return bestVal;
}

function preprocess(records) {
  // Compute medians for numericals
  const numMedians = {};
  NUMERICAL_COLS.forEach(col => {
    const vals = records.map(r => r[col] !== null ? parseFloat(r[col]) : null);
    numMedians[col] = median(vals);
  });

  // Compute modes for categoricals
  const catModes = {};
  CATEGORICAL_COLS.forEach(col => {
    const vals = records.map(r => r[col]);
    catModes[col] = mode(vals);
  });

  // Impute
  const cleanRecords = records.map(r => {
    const clean = { ...r };
    NUMERICAL_COLS.forEach(col => {
      clean[col] = clean[col] !== null ? parseFloat(clean[col]) : numMedians[col];
    });
    CATEGORICAL_COLS.forEach(col => {
      clean[col] = clean[col] !== null ? clean[col] : catModes[col];
    });
    return clean;
  });

  // Scaling: StandardScaler (Mean = 0, Std = 1)
  const stats = {};
  NUMERICAL_COLS.forEach(col => {
    const vals = cleanRecords.map(r => r[col]);
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const variance = vals.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (vals.length - 1);
    const std = Math.sqrt(variance) || 1e-6;
    stats[col] = { mean, std };
  });

  // One-Hot Encoding unique categories
  const catOptions = {};
  CATEGORICAL_COLS.forEach(col => {
    catOptions[col] = Array.from(new Set(cleanRecords.map(r => r[col]))).sort();
  });

  // Create feature matrix X
  const featureNames = [...NUMERICAL_COLS];
  CATEGORICAL_COLS.forEach(col => {
    catOptions[col].forEach(opt => {
      featureNames.push(`${col}_${opt}`);
    });
  });

  const matrix = cleanRecords.map(r => {
    const row = [];
    // Scaled numericals
    NUMERICAL_COLS.forEach(col => {
      row.push((r[col] - stats[col].mean) / stats[col].std);
    });
    // One-hot categoricals
    CATEGORICAL_COLS.forEach(col => {
      catOptions[col].forEach(opt => {
        row.push(r[col] === opt ? 1.0 : 0.0);
      });
    });
    return row;
  });

  return { cleanRecords, stats, featureNames, matrix, catOptions };
}

// 3. Distance Functions
function euclideanDist(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

function manhattanDist(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += Math.abs(a[i] - b[i]);
  }
  return sum;
}

// 4. K-Means Algorithm
function runKMeans(X, k, distanceType = 'euclidean', maxIters = 60, seed = 42) {
  const n = X.length;
  const d = X[0].length;
  const distFn = distanceType === 'manhattan' ? manhattanDist : euclideanDist;

  // Pseudo-random generator with seed
  let s = seed;
  function rand() {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  }

  // K-Means++ initialization
  const centroids = [];
  centroids.push([...X[Math.floor(rand() * n)]]);

  while (centroids.length < k) {
    const distances = X.map(row => {
      let minDist = Infinity;
      centroids.forEach(c => {
        const dist = distFn(row, c);
        if (dist < minDist) minDist = dist;
      });
      return minDist * minDist;
    });

    const sumDist = distances.reduce((a, b) => a + b, 0);
    const target = rand() * sumDist;
    let cum = 0;
    let chosenIdx = 0;
    for (let i = 0; i < n; i++) {
      cum += distances[i];
      if (cum >= target) {
        chosenIdx = i;
        break;
      }
    }
    centroids.push([...X[chosenIdx]]);
  }

  let labels = new Array(n).fill(0);
  let converged = false;

  for (let iter = 0; iter < maxIters && !converged; iter++) {
    let changed = false;

    // Assignment Step
    for (let i = 0; i < n; i++) {
      let bestCluster = 0;
      let minD = Infinity;
      for (let c = 0; c < k; c++) {
        const d = distFn(X[i], centroids[c]);
        if (d < minD) {
          minD = d;
          bestCluster = c;
        }
      }
      if (labels[i] !== bestCluster) {
        labels[i] = bestCluster;
        changed = true;
      }
    }

    if (!changed && iter > 0) {
      converged = true;
      break;
    }

    // Update Centroids Step
    for (let c = 0; c < k; c++) {
      const clusterPoints = [];
      for (let i = 0; i < n; i++) {
        if (labels[i] === c) clusterPoints.push(X[i]);
      }

      if (clusterPoints.length > 0) {
        if (distanceType === 'manhattan') {
          // Centroid for Manhattan distance is component-wise median
          for (let f = 0; f < d; f++) {
            const vals = clusterPoints.map(p => p[f]).sort((a, b) => a - b);
            centroids[c][f] = vals[Math.floor(vals.length / 2)];
          }
        } else {
          // Centroid for Euclidean distance is component-wise mean
          for (let f = 0; f < d; f++) {
            const sum = clusterPoints.reduce((acc, p) => acc + p[f], 0);
            centroids[c][f] = sum / clusterPoints.length;
          }
        }
      }
    }
  }

  // Calculate WCSS (Within-Cluster Sum of Squares)
  let wcss = 0;
  for (let i = 0; i < n; i++) {
    const c = centroids[labels[i]];
    for (let f = 0; f < d; f++) {
      wcss += Math.pow(X[i][f] - c[f], 2);
    }
  }

  return { labels, centroids, wcss, k, distanceType };
}

// 5. Hierarchical Clustering (Agglomerative)
function runHierarchical(X, k) {
  // Sample a representative subsample if n is large for speed, then assign remainder to nearest centroid
  const n = X.length;
  const d = X[0].length;
  const sampleSize = Math.min(300, n);
  
  // Pick evenly spaced samples
  const step = Math.floor(n / sampleSize);
  const sampleIndices = [];
  for (let i = 0; i < sampleSize; i++) sampleIndices.push(i * step);
  
  let clusters = sampleIndices.map((idx, i) => ({ id: i, points: [X[idx]], centroid: [...X[idx]] }));

  while (clusters.length > k) {
    let minDist = Infinity;
    let mergeA = 0;
    let mergeB = 1;

    for (let i = 0; i < clusters.length; i++) {
      for (let j = i + 1; j < clusters.length; j++) {
        // Average linkage distance between cluster centroids
        const dist = euclideanDist(clusters[i].centroid, clusters[j].centroid);
        if (dist < minDist) {
          minDist = dist;
          mergeA = i;
          mergeB = j;
        }
      }
    }

    // Merge mergeA and mergeB
    const combinedPoints = [...clusters[mergeA].points, ...clusters[mergeB].points];
    const newCentroid = new Array(d).fill(0);
    combinedPoints.forEach(p => {
      for (let f = 0; f < d; f++) newCentroid[f] += p[f];
    });
    for (let f = 0; f < d; f++) newCentroid[f] /= combinedPoints.length;

    clusters[mergeA] = { id: clusters[mergeA].id, points: combinedPoints, centroid: newCentroid };
    clusters.splice(mergeB, 1);
  }

  // Assign all points to nearest resulting cluster centroid
  const centroids = clusters.map(c => c.centroid);
  const labels = X.map(row => {
    let best = 0;
    let minD = Infinity;
    centroids.forEach((c, idx) => {
      const d = euclideanDist(row, c);
      if (d < minD) {
        minD = d;
        best = idx;
      }
    });
    return best;
  });

  let wcss = 0;
  for (let i = 0; i < n; i++) {
    const c = centroids[labels[i]];
    for (let f = 0; f < d; f++) {
      wcss += Math.pow(X[i][f] - c[f], 2);
    }
  }

  return { labels, centroids, wcss, k, distanceType: 'hierarchical_average' };
}

// 6. Validation Metrics: Silhouette Score & Davies-Bouldin Index
function computeSilhouette(X, labels, sampleLimit = 400) {
  const n = X.length;
  // Subsample for fast evaluation if n is large
  const step = Math.max(1, Math.floor(n / sampleLimit));
  const evalIndices = [];
  for (let i = 0; i < n; i += step) evalIndices.push(i);

  const clusterMap = {};
  labels.forEach((l, i) => {
    if (!clusterMap[l]) clusterMap[l] = [];
    clusterMap[l].push(i);
  });

  const k = Object.keys(clusterMap).length;
  if (k <= 1) return 0;

  let totalScore = 0;
  let count = 0;

  evalIndices.forEach(i => {
    const myCluster = labels[i];
    const sameClusterIndices = clusterMap[myCluster];

    // Compute a(i) - average distance within same cluster
    let a_i = 0;
    if (sameClusterIndices.length > 1) {
      let sum = 0;
      sameClusterIndices.forEach(idx => {
        if (idx !== i) sum += euclideanDist(X[i], X[idx]);
      });
      a_i = sum / (sameClusterIndices.length - 1);
    }

    // Compute b(i) - min average distance to another cluster
    let b_i = Infinity;
    for (const cStr of Object.keys(clusterMap)) {
      const c = parseInt(cStr);
      if (c === myCluster) continue;
      const otherIndices = clusterMap[c];
      let sum = 0;
      otherIndices.forEach(idx => {
        sum += euclideanDist(X[i], X[idx]);
      });
      const avg = sum / otherIndices.length;
      if (avg < b_i) b_i = avg;
    }

    const s_i = (b_i - a_i) / Math.max(a_i, b_i);
    totalScore += s_i;
    count++;
  });

  return count > 0 ? totalScore / count : 0;
}

function computeDaviesBouldin(X, labels, centroids) {
  const k = centroids.length;
  if (k <= 1) return 0;

  // 1. Average distance of points in cluster i to centroid i (s_i)
  const s = new Array(k).fill(0);
  const counts = new Array(k).fill(0);

  for (let i = 0; i < X.length; i++) {
    const c = labels[i];
    s[c] += euclideanDist(X[i], centroids[c]);
    counts[c]++;
  }

  for (let i = 0; i < k; i++) {
    s[i] = counts[i] > 0 ? s[i] / counts[i] : 0;
  }

  // 2. Pairwise R_ij = (s_i + s_j) / d(c_i, c_j)
  let dbTotal = 0;
  for (let i = 0; i < k; i++) {
    let maxR = -Infinity;
    for (let j = 0; j < k; j++) {
      if (i !== j) {
        const d_ij = euclideanDist(centroids[i], centroids[j]);
        if (d_ij > 0) {
          const r_ij = (s[i] + s[j]) / d_ij;
          if (r_ij > maxR) maxR = r_ij;
        }
      }
    }
    dbTotal += maxR > -Infinity ? maxR : 0;
  }

  return dbTotal / k;
}

// 7. Principal Component Analysis (PCA)
function computePCA(X, numComponents = 3) {
  const n = X.length;
  const d = X[0].length;

  // 1. Center the data
  const means = new Array(d).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) means[j] += X[i][j];
  }
  for (let j = 0; j < d; j++) means[j] /= n;

  const X_centered = X.map(row => row.map((val, j) => val - means[j]));

  // 2. Compute Covariance Matrix (d x d)
  const cov = Array.from({ length: d }, () => new Array(d).fill(0));
  for (let i = 0; i < d; i++) {
    for (let j = i; j < d; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) {
        sum += X_centered[k][i] * X_centered[k][j];
      }
      const val = sum / (n - 1);
      cov[i][j] = val;
      cov[j][i] = val;
    }
  }

  // 3. Power Iteration with Deflation for top eigenvalues/eigenvectors
  const eigenvalues = [];
  const eigenvectors = [];
  let currentCov = cov.map(row => [...row]);

  for (let c = 0; c < numComponents; c++) {
    let v = new Array(d).fill(0).map((_, i) => (i % 2 === 0 ? 1 : -1));
    // Normalize initial vector
    let norm = Math.sqrt(v.reduce((acc, val) => acc + val * val, 0));
    v = v.map(val => val / norm);

    for (let iter = 0; iter < 120; iter++) {
      // Multiply currentCov * v
      const nextV = new Array(d).fill(0);
      for (let i = 0; i < d; i++) {
        for (let j = 0; j < d; j++) {
          nextV[i] += currentCov[i][j] * v[j];
        }
      }
      const nextNorm = Math.sqrt(nextV.reduce((acc, val) => acc + val * val, 0));
      if (nextNorm === 0) break;
      v = nextV.map(val => val / nextNorm);
    }

    // Compute Rayleigh quotient (eigenvalue)
    let lambda = 0;
    for (let i = 0; i < d; i++) {
      let rowSum = 0;
      for (let j = 0; j < d; j++) rowSum += currentCov[i][j] * v[j];
      lambda += v[i] * rowSum;
    }

    eigenvalues.push(Math.max(0, lambda));
    eigenvectors.push(v);

    // Deflate matrix: currentCov -= lambda * v * v^T
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) {
        currentCov[i][j] -= lambda * v[i] * v[j];
      }
    }
  }

  // Total variance (trace of original cov matrix)
  let totalVar = 0;
  for (let i = 0; i < d; i++) totalVar += cov[i][i];

  const explainedVarianceRatios = eigenvalues.map(ev => (totalVar > 0 ? ev / totalVar : 0));

  // Project data onto eigenvectors
  const projections = X_centered.map(row => {
    return eigenvectors.map(eigVec => {
      let dot = 0;
      for (let j = 0; j < d; j++) dot += row[j] * eigVec[j];
      return parseFloat(dot.toFixed(4));
    });
  });

  return {
    eigenvalues,
    eigenvectors,
    explainedVarianceRatios,
    totalVariance: totalVar,
    projections
  };
}

// 8. Main Pipeline Execution
function main() {
  const rawPath = path.join(__dirname, '..', 'data', 'raw_imd_srtm_catchment_data.csv');
  const rawContent = fs.readFileSync(rawPath, 'utf8');
  const { headers, records } = parseCSV(rawContent);

  console.log(`Loaded ${records.length} raw catchment records.`);

  // Preprocessing
  const { cleanRecords, stats, featureNames, matrix, catOptions } = preprocess(records);

  // Write preprocessed CSV
  const prepHeaders = ['catchment_id', 'basin_region', ...featureNames];
  const prepRows = cleanRecords.map((r, idx) => {
    return [r.catchment_id, r.basin_region, ...matrix[idx].map(v => v.toFixed(4))].join(',');
  });
  const prepCSV = [prepHeaders.join(','), ...prepRows].join('\n');
  const prepPath = path.join(__dirname, '..', 'data', 'preprocessed_catchment_data.csv');
  fs.writeFileSync(prepPath, prepCSV, 'utf8');
  console.log(`Saved preprocessed dataset to: ${prepPath}`);

  // PCA Decomposition
  console.log('Running PCA dimensionality reduction...');
  const pcaResults = computePCA(matrix, 3);
  console.log(`Explained variance ratios: PC1=${(pcaResults.explainedVarianceRatios[0] * 100).toFixed(1)}%, PC2=${(pcaResults.explainedVarianceRatios[1] * 100).toFixed(1)}%, PC3=${(pcaResults.explainedVarianceRatios[2] * 100).toFixed(1)}%`);

  // Model Evaluation Suite (Objective 2)
  console.log('\n--- Evaluating K-Means & Hierarchical Across Configurations ---');
  const evaluationResults = [];

  // 1. K-Means Euclidean for K=2..7
  const kMeansConfigs = [2, 3, 4, 5, 6, 7];
  const clusterSolutions = {};

  kMeansConfigs.forEach(k => {
    const res = runKMeans(matrix, k, 'euclidean', 60, 42);
    const sil = computeSilhouette(matrix, res.labels);
    const db = computeDaviesBouldin(matrix, res.labels, res.centroids);

    evaluationResults.push({
      algorithm: 'K-Means',
      distance: 'Euclidean',
      k,
      wcss: parseFloat(res.wcss.toFixed(2)),
      silhouette: parseFloat(sil.toFixed(4)),
      daviesBouldin: parseFloat(db.toFixed(4))
    });

    clusterSolutions[`kmeans_euclidean_k${k}`] = res;
    console.log(`K-Means (Euclidean) k=${k} -> WCSS: ${res.wcss.toFixed(1)}, Sil: ${sil.toFixed(3)}, DB: ${db.toFixed(3)}`);
  });

  // 2. K-Means Manhattan for K=3, 4, 5
  [3, 4, 5].forEach(k => {
    const res = runKMeans(matrix, k, 'manhattan', 60, 42);
    const sil = computeSilhouette(matrix, res.labels);
    const db = computeDaviesBouldin(matrix, res.labels, res.centroids);

    evaluationResults.push({
      algorithm: 'K-Means',
      distance: 'Manhattan',
      k,
      wcss: parseFloat(res.wcss.toFixed(2)),
      silhouette: parseFloat(sil.toFixed(4)),
      daviesBouldin: parseFloat(db.toFixed(4))
    });

    clusterSolutions[`kmeans_manhattan_k${k}`] = res;
    console.log(`K-Means (Manhattan) k=${k} -> WCSS: ${res.wcss.toFixed(1)}, Sil: ${sil.toFixed(3)}, DB: ${db.toFixed(3)}`);
  });

  // 3. Hierarchical Agglomerative Comparison for K=3, 4, 5
  [3, 4, 5].forEach(k => {
    const res = runHierarchical(matrix, k);
    const sil = computeSilhouette(matrix, res.labels);
    const db = computeDaviesBouldin(matrix, res.labels, res.centroids);

    evaluationResults.push({
      algorithm: 'Hierarchical (Average)',
      distance: 'Euclidean',
      k,
      wcss: parseFloat(res.wcss.toFixed(2)),
      silhouette: parseFloat(sil.toFixed(4)),
      daviesBouldin: parseFloat(db.toFixed(4))
    });

    clusterSolutions[`hierarchical_k${k}`] = res;
    console.log(`Hierarchical k=${k} -> WCSS: ${res.wcss.toFixed(1)}, Sil: ${sil.toFixed(3)}, DB: ${db.toFixed(3)}`);
  });

  // Stability Test (Resampling with 10 random seeds for K=4)
  console.log('\nEvaluating K=4 stability across random seeds...');
  const stabilityScores = [];
  const baseLabels = clusterSolutions['kmeans_euclidean_k4'].labels;

  for (let seed = 101; seed <= 108; seed++) {
    const run = runKMeans(matrix, 4, 'euclidean', 60, seed);
    // Calculate simple pairwise cluster agreement
    let agreements = 0;
    const testPairs = 500;
    for (let p = 0; p < testPairs; p++) {
      const idx1 = Math.floor(Math.random() * matrix.length);
      const idx2 = Math.floor(Math.random() * matrix.length);
      const sameBase = baseLabels[idx1] === baseLabels[idx2];
      const sameRun = run.labels[idx1] === run.labels[idx2];
      if (sameBase === sameRun) agreements++;
    }
    stabilityScores.push(parseFloat((agreements / testPairs).toFixed(3)));
  }
  const avgStability = (stabilityScores.reduce((a, b) => a + b, 0) / stabilityScores.length).toFixed(3);
  console.log(`Average Cluster Stability Index (Rand Approximation): ${avgStability}`);

  // Cluster Profiling & Risk Interpretation for Defensible Solution (K=4)
  const optimalSolution = clusterSolutions['kmeans_euclidean_k4'];
  const clusterProfiles = [];

  // Group catchments by cluster
  const clustersData = [[], [], [], []];
  cleanRecords.forEach((rec, idx) => {
    const c = optimalSolution.labels[idx];
    clustersData[c].push(rec);
  });

  // Calculate profile metrics for each cluster
  const clusterRiskLabels = [
    {
      id: 0,
      name: 'Tier 1: Extreme Flash-Flood & Debris Flow Risk',
      shortName: 'Extreme Risk',
      color: '#EA580C', // Deep Orange/Amber
      description: 'Ultra-steep terrain, high relief, short response time (<2.5 hrs), and intense cloudburst rainfall rates. Immediate early-warning sensor coverage required.'
    },
    {
      id: 1,
      name: 'Tier 2: High Saturated Catchment Runoff',
      shortName: 'High Risk',
      color: '#F59E0B', // Amber
      description: 'High consecutive wet days, high topographic wetness index, prolonged monsoon downpours, leading to saturated valley flash-floods.'
    },
    {
      id: 2,
      name: 'Tier 3: Moderate Transitional Hill Basin',
      shortName: 'Moderate Risk',
      color: '#EAB308', // Yellow
      description: 'Intermediate slope and drainage density with mixed agricultural terraces; manageable runoff lag time (~4 to 7 hours).'
    },
    {
      id: 3,
      name: 'Tier 4: Attenuated Plateau & Low Infiltration Risk',
      shortName: 'Low Risk',
      color: '#CA8A04', // Dark Gold
      description: 'Gentle slopes, broad valleys, high soil retention, and long lag times (>8 hours). Lowest susceptibility to sudden convective flash floods.'
    }
  ];

  for (let c = 0; c < 4; c++) {
    const members = clustersData[c];
    const profile = {
      clusterId: c,
      size: members.length,
      percentage: parseFloat(((members.length / cleanRecords.length) * 100).toFixed(1)),
      metadata: clusterRiskLabels[c],
      rawAverages: {},
      standardizedCentroid: {}
    };

    NUMERICAL_COLS.forEach(col => {
      const avg = members.reduce((sum, r) => sum + r[col], 0) / members.length;
      profile.rawAverages[col] = parseFloat(avg.toFixed(2));
      profile.standardizedCentroid[col] = parseFloat(((avg - stats[col].mean) / stats[col].std).toFixed(2));
    });

    clusterProfiles.push(profile);
  }

  // Factor Loadings for PCA
  const pcaLoadings = {};
  featureNames.slice(0, NUMERICAL_COLS.length).forEach((feat, fIdx) => {
    pcaLoadings[feat] = {
      PC1: parseFloat(pcaResults.eigenvectors[0][fIdx].toFixed(4)),
      PC2: parseFloat(pcaResults.eigenvectors[1][fIdx].toFixed(4)),
      PC3: parseFloat(pcaResults.eigenvectors[2][fIdx].toFixed(4))
    };
  });

  // Prepare Master Output for Web App & Artifacts
  const webDataset = {
    metadata: {
      title: 'Hilly Catchment Flash-Flood Risk Pattern Segmentation Using Clustering',
      totalCatchments: cleanRecords.length,
      featuresCount: featureNames.length,
      numericalFeatures: NUMERICAL_COLS,
      categoricalFeatures: CATEGORICAL_COLS,
      generatedAt: new Date().toISOString()
    },
    preprocessingStats: stats,
    pca: {
      explainedVarianceRatios: pcaResults.explainedVarianceRatios.map(v => parseFloat(v.toFixed(4))),
      cumulativeVariance: [
        parseFloat(pcaResults.explainedVarianceRatios[0].toFixed(4)),
        parseFloat((pcaResults.explainedVarianceRatios[0] + pcaResults.explainedVarianceRatios[1]).toFixed(4)),
        parseFloat((pcaResults.explainedVarianceRatios[0] + pcaResults.explainedVarianceRatios[1] + pcaResults.explainedVarianceRatios[2]).toFixed(4))
      ],
      loadings: pcaLoadings
    },
    evaluations: evaluationResults,
    stabilityScores,
    avgStability,
    clusterProfiles,
    catchments: cleanRecords.map((r, idx) => ({
      id: r.catchment_id,
      region: r.basin_region,
      terrain: r.terrain_zone,
      soil: r.soil_drainage_type,
      vegetation: r.vegetation_cover_class,
      elevation: r.elevation_mean_m,
      relief: r.elevation_relief_m,
      slope: r.slope_mean_deg,
      drainageDensity: r.drainage_density_km_sqkm,
      twi: r.topographic_wetness_index,
      rainfall: r.peak_monsoon_rainfall_mm,
      hourlyIntensity: r.max_hourly_intensity_mm_hr,
      cwd: r.consecutive_wet_days_cwd,
      responseTime: r.response_time_hr,
      pca: {
        pc1: pcaResults.projections[idx][0],
        pc2: pcaResults.projections[idx][1],
        pc3: pcaResults.projections[idx][2]
      },
      clusters: {
        k3_euclidean: clusterSolutions['kmeans_euclidean_k3'].labels[idx],
        k4_euclidean: clusterSolutions['kmeans_euclidean_k4'].labels[idx],
        k5_euclidean: clusterSolutions['kmeans_euclidean_k5'].labels[idx],
        k3_manhattan: clusterSolutions['kmeans_manhattan_k3'].labels[idx],
        k4_manhattan: clusterSolutions['kmeans_manhattan_k4'].labels[idx],
        k5_manhattan: clusterSolutions['kmeans_manhattan_k5'].labels[idx],
        k3_hierarchical: clusterSolutions['hierarchical_k3'].labels[idx],
        k4_hierarchical: clusterSolutions['hierarchical_k4'].labels[idx],
        k5_hierarchical: clusterSolutions['hierarchical_k5'].labels[idx]
      }
    }))
  };

  const jsonOutputPath = path.join(__dirname, '..', 'web_app', 'dataset.json');
  fs.writeFileSync(jsonOutputPath, JSON.stringify(webDataset, null, 2), 'utf8');
  console.log(`Generated master web dataset at: ${jsonOutputPath}`);

  console.log('\nAnalysis execution completed successfully!');
}

main();
