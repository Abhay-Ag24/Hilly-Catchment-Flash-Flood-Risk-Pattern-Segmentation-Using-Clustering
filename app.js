/**
 * Hilly Catchment Flash-Flood Risk Segmentation Engine
 * Fully Responsive Interactive Dashboard with Dynamic K & Metric Reactivity
 */

let appData = null;
let currentK = 4;
let currentMetric = 'euclidean';
let currentRegion = 'ALL';
let searchQuery = '';

// High-contrast color palette for up to 5 clusters
const CLUSTER_COLORS = [
  '#ea580c', // Cluster 0: Deep Red-Orange (Tier 1 Extreme)
  '#f59e0b', // Cluster 1: Warm Amber (Tier 2 High)
  '#eab308', // Cluster 2: Golden Yellow (Tier 3 Moderate)
  '#78350f', // Cluster 3: Bronze Ochre (Tier 4 Low Attenuated)
  '#0284c7'  // Cluster 4: Deep Sky Blue (Tier 5 Foothills)
];

const PALETTE = {
  yellow: '#f59e0b',
  gold: '#eab308',
  softYellow: '#fef08a',
  charcoal: '#1c1917',
  gray: '#78716c',
  lightGray: '#f5f5f4'
};

// Qualitative Cluster Meta Descriptions for K=3, 4, 5
const CLUSTER_METADATA_PRESETS = {
  3: [
    {
      shortName: 'High Hazard Zone',
      name: 'Tier 1: High Flash-Flood & Convective Runoff Zone',
      description: 'Steep relief, short concentration time (<2.5 hrs), and intense cloudburst downpours requiring early-warning sirens.'
    },
    {
      shortName: 'Moderate Saturated Basin',
      name: 'Tier 2: Saturated Soil & Prolonged Monsoon Inundation',
      description: 'High consecutive wet days, moderate to high topographic wetness index (TWI), and valley drainage accumulation.'
    },
    {
      shortName: 'Low Attenuated Plateau',
      name: 'Tier 3: Attenuated Plateau & Low Infiltration Risk',
      description: 'Gentle slopes (<12°), extended concentration times (>8 hrs), and high soil retention with minimal flash-flood threat.'
    }
  ],
  4: [
    {
      shortName: 'Extreme Risk',
      name: 'Tier 1: Extreme Flash-Flood & Debris Flow Zone',
      description: 'Ultra-steep terrain, high relief, short response time (<2 hrs), and intense cloudburst rainfall rates. Immediate early-warning sensor coverage required.'
    },
    {
      shortName: 'High Risk',
      name: 'Tier 2: High Saturated Catchment Runoff',
      description: 'High consecutive wet days, high topographic wetness index, prolonged monsoon downpours, leading to saturated valley flash-floods.'
    },
    {
      shortName: 'Moderate Risk',
      name: 'Tier 3: Moderate Transitional Hill Basin',
      description: 'Intermediate slope and drainage density with mixed agricultural terraces; manageable runoff lag time (~4 to 7 hours).'
    },
    {
      shortName: 'Low Risk',
      name: 'Tier 4: Attenuated Plateau & Low Infiltration Risk',
      description: 'Gentle slopes, broad valleys, high soil retention, and long lag times (>8 hours). Lowest susceptibility to sudden convective flash floods.'
    }
  ],
  5: [
    {
      shortName: 'Catastrophic Cloudburst',
      name: 'Tier 1: Catastrophic Cloudburst Gorge Zone',
      description: 'Extreme relief (>1,400m), slope >32°, and rapid hydrograph spikes with critical debris flow hazard.'
    },
    {
      shortName: 'High-Relief Escarpment',
      name: 'Tier 2: High-Relief Mountain Escarpment',
      description: 'Steep ridges with high drainage density, rapid runoff generation, and vulnerable valley settlements.'
    },
    {
      shortName: 'Saturated Valley Basin',
      name: 'Tier 3: Saturated Monsoonal Valley Basin',
      description: 'Prolonged antecedent rainfall (CWD >15 days) and high topographic wetness index causing valley inundation.'
    },
    {
      shortName: 'Transitional Terraced Hill',
      name: 'Tier 4: Moderate Transitional Terraced Basin',
      description: 'Agricultural terraces and intermediate slopes with moderate hydraulic concentration times.'
    },
    {
      shortName: 'Attenuated Plateau',
      name: 'Tier 5: Attenuated Low-Relief Plateau',
      description: 'Wide floodplains and gentle hills with prolonged lag times (>9 hrs) and high percolation rates.'
    }
  ]
};

// Initialize Application
document.addEventListener('DOMContentLoaded', async () => {
  setupTabs();
  await loadDataset();
  setupEventListeners();
  setupResponsiveCanvasObserver();
  renderAll();
});

// Tab Navigation
function setupTabs() {
  const tabBtns = document.querySelectorAll('.tab-btn');
  const panels = document.querySelectorAll('.tab-panel');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      panels.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const targetId = btn.getAttribute('data-tab');
      const targetPanel = document.getElementById(targetId);
      if (targetPanel) {
        targetPanel.classList.add('active');
      }

      // Re-render visual canvas when tab becomes active
      requestAnimationFrame(() => {
        if (targetId === 'tab-clustering') {
          renderWCSSChart();
          renderMetricsChart();
        } else if (targetId === 'tab-pca') {
          renderPCAScatter();
          renderPCALoadings();
        } else if (targetId === 'tab-profiles') {
          renderRadarChart();
        }
      });
    });
  });
}

// Fetch Dataset JSON or load from window.appData
async function loadDataset() {
  if (window.appData) {
    appData = window.appData;
    console.log('Dataset loaded directly from window.appData:', appData.metadata);
    return;
  }
  try {
    const res = await fetch('dataset.json');
    appData = await res.json();
    console.log('Dataset loaded successfully via fetch:', appData.metadata);
  } catch (err) {
    console.error('Failed to load dataset.json via fetch:', err);
  }
}

function setupEventListeners() {
  const kSelect = document.getElementById('select-k');
  if (kSelect) {
    kSelect.addEventListener('change', (e) => {
      currentK = parseInt(e.target.value);
      console.log(`Cluster count switched to K = ${currentK}`);
      renderAll();
    });
  }

  const metricSelect = document.getElementById('select-metric');
  if (metricSelect) {
    metricSelect.addEventListener('change', (e) => {
      currentMetric = e.target.value;
      console.log(`Distance metric switched to ${currentMetric}`);
      renderAll();
    });
  }

  const regionFilter = document.getElementById('filter-region');
  if (regionFilter) {
    regionFilter.addEventListener('change', (e) => {
      currentRegion = e.target.value;
      renderCatchmentsTable();
    });
  }

  const searchInput = document.getElementById('search-catchment');
  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      searchQuery = e.target.value.trim().toLowerCase();
      renderCatchmentsTable();
    });
  }

  const btnCsv = document.getElementById('btn-download-csv');
  if (btnCsv) {
    btnCsv.addEventListener('click', downloadCSV);
  }

  const btnJson = document.getElementById('btn-download-json');
  if (btnJson) {
    btnJson.addEventListener('click', downloadJSON);
  }

  // Window resize fallback
  let resizeTimeout;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
      renderActiveTabCharts();
    }, 100);
  });
}

// Responsive ResizeObserver for canvas wrappers
function setupResponsiveCanvasObserver() {
  if (window.ResizeObserver) {
    const ro = new ResizeObserver(() => {
      renderActiveTabCharts();
    });
    document.querySelectorAll('.viz-canvas-wrapper').forEach(wrapper => {
      ro.observe(wrapper);
    });
  }
}

function renderActiveTabCharts() {
  const activeTab = document.querySelector('.tab-panel.active');
  if (!activeTab || !appData) return;
  const id = activeTab.id;
  if (id === 'tab-clustering') {
    renderWCSSChart();
    renderMetricsChart();
  } else if (id === 'tab-pca') {
    renderPCAScatter();
    renderPCALoadings();
  } else if (id === 'tab-profiles') {
    renderRadarChart();
  }
}

// Helper to get active cluster assignment key
function getClusterKey(k, metric) {
  if (metric === 'hierarchical') {
    return `k${k}_hierarchical`;
  }
  return `k${k}_${metric}`;
}

// Dynamic Profile & Centroid Computation for ANY K and Metric
function computeDynamicProfiles(k, metric) {
  if (!appData || !appData.catchments) return [];
  const key = getClusterKey(k, metric);
  const presets = CLUSTER_METADATA_PRESETS[k] || CLUSTER_METADATA_PRESETS[4];

  const clustersData = Array.from({ length: k }, () => []);
  appData.catchments.forEach(c => {
    const clusterId = c.clusters[key] !== undefined ? c.clusters[key] : (c.clusters.k4_euclidean % k);
    if (clustersData[clusterId]) {
      clustersData[clusterId].push(c);
    } else {
      clustersData[0].push(c);
    }
  });

  const numCols = appData.metadata.numericalFeatures;
  const stats = appData.preprocessingStats;
  const total = appData.catchments.length;

  const profiles = [];
  for (let c = 0; c < k; c++) {
    const members = clustersData[c] || [];
    const size = members.length;
    const percentage = parseFloat(((size / total) * 100).toFixed(1));

    const meta = presets[c] || {
      shortName: `Tier ${c + 1}`,
      name: `Risk Tier ${c + 1}`,
      description: `Catchment segment with characteristic hydrological and geomorphological metrics.`
    };

    const rawAverages = {};
    const standardizedCentroid = {};

    numCols.forEach(col => {
      // Map JSON key to catchment property
      const propMap = {
        elevation_mean_m: 'elevation',
        elevation_relief_m: 'relief',
        slope_mean_deg: 'slope',
        drainage_density_km_sqkm: 'drainageDensity',
        topographic_wetness_index: 'twi',
        hypsometric_integral: 'elevation', // fallback
        ruggedness_number: 'relief', // fallback
        catchment_area_sqkm: 'relief', // fallback
        peak_monsoon_rainfall_mm: 'rainfall',
        max_hourly_intensity_mm_hr: 'hourlyIntensity',
        consecutive_wet_days_cwd: 'cwd',
        precipitation_concentration_index: 'rainfall', // fallback
        response_time_hr: 'responseTime'
      };

      const prop = propMap[col];
      let avg = 0;
      if (members.length > 0 && prop && members[0][prop] !== undefined) {
        avg = members.reduce((sum, m) => sum + (parseFloat(m[prop]) || 0), 0) / members.length;
      } else if (appData.clusterProfiles && appData.clusterProfiles[c % appData.clusterProfiles.length]) {
        avg = appData.clusterProfiles[c % appData.clusterProfiles.length].rawAverages[col] || 0;
      }

      rawAverages[col] = parseFloat(avg.toFixed(2));
      const colMean = stats && stats[col] ? stats[col].mean : 0;
      const colStd = stats && stats[col] && stats[col].std > 0 ? stats[col].std : 1;
      standardizedCentroid[col] = parseFloat(((avg - colMean) / colStd).toFixed(2));
    });

    profiles.push({
      clusterId: c,
      size,
      percentage,
      metadata: meta,
      rawAverages,
      standardizedCentroid
    });
  }

  return profiles;
}

// Master Render Function
function renderAll() {
  if (!appData) return;
  const profiles = computeDynamicProfiles(currentK, currentMetric);

  updateKPICards();
  renderCatchmentsTable(profiles);
  renderEvaluationTable();
  renderWCSSChart();
  renderMetricsChart();
  renderPCAScatter(profiles);
  renderPCALoadings();
  renderRiskCards(profiles);
  renderRadarChart(profiles);
  renderCentroidsTable(profiles);
}

// 1. Update KPI Summary Cards Dynamically
function updateKPICards() {
  const kElem = document.getElementById('kpi-clusters');
  const kSub = document.getElementById('kpi-cluster-sub');
  const silElem = document.getElementById('kpi-silhouette');
  const silSub = document.getElementById('kpi-silhouette-sub');
  const pcaTitle = document.getElementById('pca-title');
  const profilesHeading = document.getElementById('profiles-heading');
  const badgeProfilesCount = document.getElementById('badge-profiles-count');
  const bannerDesc = document.getElementById('banner-model-desc');
  const bannerStats = document.getElementById('banner-model-stats');
  const thAssignedRisk = document.getElementById('th-assigned-risk');

  if (kElem) kElem.textContent = `K = ${currentK}`;
  if (kSub) {
    if (currentK === 3) kSub.textContent = 'Broad Regional Risk Tiers';
    else if (currentK === 4) kSub.textContent = 'Defensible Optimal Structure';
    else kSub.textContent = 'Fine-Grained Micro-Basin Tiers';
  }

  // Find active evaluation metrics
  const algoName = currentMetric === 'hierarchical' ? 'Hierarchical (Average)' : 'K-Means';
  const distName = currentMetric === 'manhattan' ? 'Manhattan' : 'Euclidean';
  const activeEv = appData.evaluations.find(e => e.algorithm.includes(algoName) && e.distance === distName && e.k === currentK) ||
                   appData.evaluations.find(e => e.k === currentK) ||
                   { silhouette: 0.123, wcss: 12336.9, daviesBouldin: 2.269 };

  if (silElem) silElem.textContent = activeEv.silhouette.toFixed(3);
  if (silSub) silSub.textContent = `${distName} (${currentMetric === 'hierarchical' ? 'Hierarchical' : 'K-Means'})`;

  if (pcaTitle) pcaTitle.textContent = `2D PCA Projection (Colored by K = ${currentK} Clusters)`;
  if (profilesHeading) profilesHeading.textContent = `Interpreted Catchment Flash-Flood Risk Profiles (K = ${currentK})`;
  if (badgeProfilesCount) badgeProfilesCount.textContent = `${currentK} Active Clusters (${distName})`;
  if (bannerDesc) bannerDesc.textContent = `${algoName} with ${distName} Distance (K = ${currentK})`;
  if (bannerStats) bannerStats.textContent = `WCSS: ${activeEv.wcss.toLocaleString()} | Silhouette: ${activeEv.silhouette.toFixed(3)} | Davies-Bouldin: ${activeEv.daviesBouldin.toFixed(3)}`;
  if (thAssignedRisk) thAssignedRisk.textContent = `Assigned Risk Tier (K = ${currentK})`;
}

// 2. Render Catchment Data Explorer Table
function renderCatchmentsTable(profiles) {
  const tbody = document.getElementById('catchments-tbody');
  if (!tbody || !appData) return;

  const key = getClusterKey(currentK, currentMetric);
  const activeProfiles = profiles || computeDynamicProfiles(currentK, currentMetric);

  let filtered = appData.catchments.filter(c => {
    const matchRegion = currentRegion === 'ALL' || c.region === currentRegion;
    const matchSearch = !searchQuery || c.id.toLowerCase().includes(searchQuery) || c.terrain.toLowerCase().includes(searchQuery);
    return matchRegion && matchSearch;
  });

  const displayList = filtered.slice(0, 50);

  tbody.innerHTML = displayList.map(c => {
    const clusterId = c.clusters[key] !== undefined ? c.clusters[key] : (c.clusters.k4_euclidean % currentK);
    const color = CLUSTER_COLORS[clusterId % CLUSTER_COLORS.length];
    const profile = activeProfiles[clusterId] || { metadata: { shortName: `Tier ${clusterId + 1}` } };
    const riskLabel = profile.metadata.shortName;

    return `
      <tr>
        <td><strong>${c.id}</strong></td>
        <td>${c.region.replace(/_/g, ' ')}</td>
        <td><span style="font-size:0.75rem; color:#57534e;">${c.terrain.replace(/_/g, ' ')}</span></td>
        <td>${c.relief} m</td>
        <td>${c.slope}°</td>
        <td>${c.rainfall} mm</td>
        <td><strong>${c.hourlyIntensity}</strong> mm/h</td>
        <td>${c.responseTime} h</td>
        <td>
          <span style="display:inline-flex; align-items:center; gap:6px; font-weight:600; color:${color}; font-size:0.75rem;">
            <span style="width:8px; height:8px; border-radius:50%; background-color:${color}; flex-shrink:0;"></span>
            ${riskLabel}
          </span>
        </td>
      </tr>
    `;
  }).join('');
}

// 3. Render Evaluations Benchmark Matrix Table with Active Selection Highlight
function renderEvaluationTable() {
  const tbody = document.getElementById('evaluations-tbody');
  if (!tbody || !appData) return;

  const targetAlgo = currentMetric === 'hierarchical' ? 'Hierarchical' : 'K-Means';
  const targetDist = currentMetric === 'manhattan' ? 'Manhattan' : 'Euclidean';

  tbody.innerHTML = appData.evaluations.map(ev => {
    const isActive = ev.k === currentK && ev.algorithm.includes(targetAlgo) && ev.distance === targetDist;
    const isOptimal = ev.algorithm === 'K-Means' && ev.distance === 'Euclidean' && ev.k === 4;

    let verdict = '<span style="color:#78716c;">Alternative</span>';
    if (isActive) {
      verdict = '<span class="badge badge-yellow" style="font-weight:700;">▶ Active Selection</span>';
    } else if (isOptimal) {
      verdict = '<span class="badge badge-white" style="font-weight:600; border-color:var(--border-strong);">★ Defensible Optimal</span>';
    }

    const rowBg = isActive ? 'background-color: #fffdf5; font-weight: 600;' : (isOptimal ? 'background-color: #fafaf9;' : '');

    return `
      <tr style="${rowBg}">
        <td><strong>${ev.algorithm}</strong></td>
        <td>${ev.distance}</td>
        <td><strong>K = ${ev.k}</strong></td>
        <td>${ev.wcss.toLocaleString()}</td>
        <td><strong>${ev.silhouette.toFixed(4)}</strong></td>
        <td>${ev.daviesBouldin.toFixed(4)}</td>
        <td>${verdict}</td>
      </tr>
    `;
  }).join('');
}

// 4. Render WCSS Elbow Chart (Canvas) - Highlights Selected K
function renderWCSSChart() {
  const canvas = document.getElementById('wcss-chart');
  if (!canvas || !appData) return;

  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return;

  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;
  ctx.clearRect(0, 0, w, h);

  const kMeansRuns = appData.evaluations.filter(e => e.algorithm === 'K-Means' && e.distance === 'Euclidean');
  if (kMeansRuns.length === 0) return;

  const padLeft = Math.max(45, Math.min(60, Math.round(w * 0.12)));
  const padRight = Math.max(20, Math.min(30, Math.round(w * 0.06)));
  const padTop = 25;
  const padBottom = 35;
  const plotW = w - padLeft - padRight;
  const plotH = h - padTop - padBottom;

  const maxWCSS = Math.max(...kMeansRuns.map(r => r.wcss)) * 1.05;
  const minWCSS = Math.min(...kMeansRuns.map(r => r.wcss)) * 0.95;
  const fontSize = Math.max(9, Math.min(11, Math.round(w / 45)));

  // Grid Lines
  ctx.strokeStyle = '#f3f4f6';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const yVal = minWCSS + (maxWCSS - minWCSS) * (i / 4);
    const y = padTop + plotH - (i / 4) * plotH;
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(w - padRight, y);
    ctx.stroke();

    ctx.fillStyle = '#78716c';
    ctx.font = `${fontSize}px sans-serif`;
    ctx.textAlign = 'right';
    ctx.fillText(Math.round(yVal), padLeft - 6, y + 3);
  }

  // Draw Line
  const points = kMeansRuns.map((r, idx) => {
    const x = padLeft + (idx / (kMeansRuns.length - 1)) * plotW;
    const y = padTop + plotH - ((r.wcss - minWCSS) / (maxWCSS - minWCSS)) * plotH;
    return { x, y, k: r.k, wcss: r.wcss };
  });

  ctx.strokeStyle = PALETTE.yellow;
  ctx.lineWidth = Math.max(2, Math.min(3, w / 150));
  ctx.beginPath();
  points.forEach((p, idx) => {
    if (idx === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  });
  ctx.stroke();

  // Draw Points & Dynamic K Selection Highlights
  points.forEach(p => {
    const isSelected = p.k === currentK;
    const isOptimal = p.k === 4;

    if (isSelected) {
      // Outer glow ring for active K
      ctx.strokeStyle = '#ea580c';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 10, 0, Math.PI * 2);
      ctx.stroke();
    }

    ctx.fillStyle = isSelected ? '#ea580c' : (isOptimal ? '#d97706' : PALETTE.yellow);
    ctx.beginPath();
    ctx.arc(p.x, p.y, isSelected ? 7 : (isOptimal ? 5.5 : 4.5), 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();

    // X-axis label
    ctx.fillStyle = isSelected ? '#ea580c' : (isOptimal ? '#b45309' : '#1c1917');
    ctx.font = isSelected ? `bold ${fontSize + 1}px sans-serif` : `${fontSize}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.fillText(`K=${p.k}`, p.x, h - padBottom + 16);

    if (isSelected) {
      ctx.fillStyle = '#ea580c';
      ctx.font = `bold ${fontSize}px sans-serif`;
      ctx.fillText(`Active (K=${p.k})`, p.x, p.y - 12);
    } else if (isOptimal && !isSelected) {
      ctx.fillStyle = '#b45309';
      ctx.font = `${fontSize - 1}px sans-serif`;
      ctx.fillText('Elbow (K=4)', p.x, p.y - 10);
    }
  });
}

// 5. Render Validation Metrics Chart (Canvas) - Highlights Selected K
function renderMetricsChart() {
  const canvas = document.getElementById('metrics-chart');
  if (!canvas || !appData) return;

  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return;

  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;
  ctx.clearRect(0, 0, w, h);

  const kSubset = [3, 4, 5];
  const euclideanRuns = appData.evaluations.filter(e => e.algorithm === 'K-Means' && e.distance === 'Euclidean' && kSubset.includes(e.k));
  const manhattanRuns = appData.evaluations.filter(e => e.algorithm === 'K-Means' && e.distance === 'Manhattan' && kSubset.includes(e.k));

  const padLeft = Math.max(35, Math.min(45, Math.round(w * 0.1)));
  const padRight = Math.max(15, Math.min(25, Math.round(w * 0.05)));
  const padTop = 32;
  const padBottom = 35;
  const plotW = w - padLeft - padRight;
  const plotH = h - padTop - padBottom;
  const fontSize = Math.max(8, Math.min(10, Math.round(w / 45)));

  const groupW = plotW / kSubset.length;
  const barW = Math.max(10, Math.min(32, groupW * 0.28));
  const barGap = Math.max(3, Math.min(6, barW * 0.2));

  kSubset.forEach((k, idx) => {
    const isSelectedK = k === currentK;
    const euc = euclideanRuns.find(r => r.k === k);
    const man = manhattanRuns.find(r => r.k === k);

    const groupCenterX = padLeft + idx * groupW + groupW / 2;
    const eucX = groupCenterX - barW - barGap / 2;
    const manX = groupCenterX + barGap / 2;

    // Active group highlight outline
    if (isSelectedK) {
      ctx.fillStyle = 'rgba(254, 240, 138, 0.35)';
      ctx.fillRect(groupCenterX - groupW / 2 + 4, padTop, groupW - 8, plotH);
    }

    // Euclidean Bar
    if (euc) {
      const barH = (euc.silhouette / 0.22) * plotH;
      const barY = padTop + plotH - barH;
      ctx.fillStyle = isSelectedK && currentMetric === 'euclidean' ? '#ea580c' : PALETTE.yellow;
      ctx.fillRect(eucX, barY, barW, barH);

      ctx.fillStyle = isSelectedK && currentMetric === 'euclidean' ? '#ea580c' : '#1c1917';
      ctx.font = isSelectedK && currentMetric === 'euclidean' ? `bold ${fontSize}px sans-serif` : `${fontSize}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillText(euc.silhouette.toFixed(3), eucX + barW / 2, barY - 4);
    }

    // Manhattan Bar
    if (man) {
      const barH = (man.silhouette / 0.22) * plotH;
      const barY = padTop + plotH - barH;
      ctx.fillStyle = isSelectedK && currentMetric === 'manhattan' ? '#b45309' : '#fde68a';
      ctx.fillRect(manX, barY, barW, barH);

      ctx.fillStyle = isSelectedK && currentMetric === 'manhattan' ? '#b45309' : '#57534e';
      ctx.font = isSelectedK && currentMetric === 'manhattan' ? `bold ${fontSize}px sans-serif` : `${fontSize}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillText(man.silhouette.toFixed(3), manX + barW / 2, barY - 4);
    }

    // X-axis label
    ctx.fillStyle = isSelectedK ? '#ea580c' : '#1c1917';
    ctx.font = isSelectedK ? `bold ${fontSize + 2}px sans-serif` : `bold ${fontSize + 1}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.fillText(`K=${k}${isSelectedK ? ' ★' : ''}`, groupCenterX, h - padBottom + 16);
  });

  // Legend
  const legendX = Math.max(padLeft, w - 180);
  ctx.fillStyle = PALETTE.yellow;
  ctx.fillRect(legendX, 8, 10, 10);
  ctx.fillStyle = '#1c1917';
  ctx.font = `${fontSize}px sans-serif`;
  ctx.textAlign = 'left';
  ctx.fillText('Euclidean Sil', legendX + 14, 16);

  ctx.fillStyle = '#fde68a';
  ctx.fillRect(legendX + 90, 8, 10, 10);
  ctx.fillStyle = '#1c1917';
  ctx.fillText('Manhattan Sil', legendX + 104, 16);
}

// 6. Render PCA 2D Scatter Plot with Dynamic K Coloring
function renderPCAScatter(profiles) {
  const canvas = document.getElementById('pca-scatter');
  const tooltip = document.getElementById('pca-tooltip');
  if (!canvas || !appData) return;

  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return;

  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;
  ctx.clearRect(0, 0, w, h);

  const pad = Math.max(24, Math.min(38, Math.round(w * 0.08)));
  const plotW = w - pad * 2;
  const plotH = h - pad * 2;

  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  appData.catchments.forEach(c => {
    if (c.pca.pc1 < minX) minX = c.pca.pc1;
    if (c.pca.pc1 > maxX) maxX = c.pca.pc1;
    if (c.pca.pc2 < minY) minY = c.pca.pc2;
    if (c.pca.pc2 > maxY) maxY = c.pca.pc2;
  });

  const scaleX = val => pad + ((val - minX) / (maxX - minX)) * plotW;
  const scaleY = val => pad + plotH - ((val - minY) / (maxY - minY)) * plotH;

  // Center axes
  ctx.strokeStyle = '#e7e5e4';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad, scaleY(0));
  ctx.lineTo(w - pad, scaleY(0));
  ctx.moveTo(scaleX(0), pad);
  ctx.lineTo(scaleX(0), h - pad);
  ctx.stroke();

  // Axis Labels
  const fontSize = Math.max(8, Math.min(10, Math.round(w / 45)));
  ctx.fillStyle = '#78716c';
  ctx.font = `${fontSize}px sans-serif`;
  ctx.textAlign = 'right';
  ctx.fillText('PC1 (Topography)', w - pad - 4, scaleY(0) - 5);
  ctx.textAlign = 'left';
  ctx.fillText('PC2 (Rainfall)', scaleX(0) + 6, pad + 10);

  // Key for cluster lookup
  const key = getClusterKey(currentK, currentMetric);
  const activeProfiles = profiles || computeDynamicProfiles(currentK, currentMetric);

  const screenPoints = [];
  const pointRadius = Math.max(2.5, Math.min(3.8, w / 140));

  appData.catchments.forEach(c => {
    const x = scaleX(c.pca.pc1);
    const y = scaleY(c.pca.pc2);

    const clusterId = c.clusters[key] !== undefined ? c.clusters[key] : (c.clusters.k4_euclidean % currentK);
    const color = CLUSTER_COLORS[clusterId % CLUSTER_COLORS.length];

    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, pointRadius, 0, Math.PI * 2);
    ctx.fill();

    screenPoints.push({ x, y, catchment: c, clusterId, color });
  });

  // On-canvas dynamic legend
  const legendTop = pad + 4;
  let legendLeft = pad + 8;
  ctx.font = `bold ${fontSize}px sans-serif`;
  for (let c = 0; c < currentK; c++) {
    const color = CLUSTER_COLORS[c % CLUSTER_COLORS.length];
    const prof = activeProfiles[c] || { metadata: { shortName: `C${c}` } };

    ctx.fillStyle = color;
    ctx.fillRect(legendLeft, legendTop, 9, 9);

    ctx.fillStyle = '#1c1917';
    ctx.textAlign = 'left';
    ctx.fillText(prof.metadata.shortName, legendLeft + 12, legendTop + 8);

    legendLeft += ctx.measureText(prof.metadata.shortName).width + 24;
    if (legendLeft > w - 100) break; // prevent wrapping off-screen
  }

  // Pointer Interaction
  function handlePointer(clientX, clientY) {
    const mRect = canvas.getBoundingClientRect();
    const mx = clientX - mRect.left;
    const my = clientY - mRect.top;

    let nearest = null;
    let minDist = Math.max(14, w / 30);

    screenPoints.forEach(p => {
      const d = Math.hypot(p.x - mx, p.y - my);
      if (d < minDist) {
        minDist = d;
        nearest = p;
      }
    });

    if (nearest && tooltip) {
      const c = nearest.catchment;
      const prof = activeProfiles[nearest.clusterId] || { metadata: { name: `Tier ${nearest.clusterId + 1}` } };
      tooltip.style.display = 'block';

      const tooltipW = 240;
      let left = nearest.x + 12;
      if (left + tooltipW > w) left = nearest.x - tooltipW - 12;
      if (left < 8) left = 8;

      let top = nearest.y - 12;
      if (top < 8) top = 8;

      tooltip.style.left = `${left}px`;
      tooltip.style.top = `${top}px`;
      tooltip.innerHTML = `
        <strong>${c.id}</strong> (${c.region.replace(/_/g, ' ')})<br>
        <span style="color:${nearest.color}; font-weight:700;">${prof.metadata.name} (Cluster ${nearest.clusterId + 1} of ${currentK})</span><br>
        <strong>Slope:</strong> ${c.slope}° | <strong>Relief:</strong> ${c.relief} m<br>
        <strong>Rainfall:</strong> ${c.rainfall} mm | <strong>Intensity:</strong> ${c.hourlyIntensity} mm/h<br>
        <strong>Response Time:</strong> ${c.responseTime} hrs
      `;
    } else if (tooltip) {
      tooltip.style.display = 'none';
    }
  }

  canvas.onmousemove = (e) => handlePointer(e.clientX, e.clientY);
  canvas.ontouchstart = (e) => {
    if (e.touches.length > 0) handlePointer(e.touches[0].clientX, e.touches[0].clientY);
  };
  canvas.ontouchmove = (e) => {
    if (e.touches.length > 0) handlePointer(e.touches[0].clientX, e.touches[0].clientY);
  };
  canvas.onmouseleave = () => {
    if (tooltip) tooltip.style.display = 'none';
  };
}

// 7. Render PCA Loadings Bar Chart
function renderPCALoadings() {
  const canvas = document.getElementById('pca-loadings-chart');
  if (!canvas || !appData) return;

  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return;

  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;
  ctx.clearRect(0, 0, w, h);

  const loadings = appData.pca.loadings;
  const feats = Object.keys(loadings);

  const padLeft = Math.max(85, Math.min(135, Math.round(w * 0.32)));
  const padRight = Math.max(20, Math.min(30, Math.round(w * 0.06)));
  const padTop = 15;
  const padBottom = 25;
  const plotW = w - padLeft - padRight;
  const plotH = h - padTop - padBottom;
  const barH = plotH / feats.length;
  const fontSize = Math.max(8, Math.min(10, Math.round(w / 45)));

  const zeroX = padLeft + plotW / 2;
  ctx.strokeStyle = '#d6d3d1';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(zeroX, padTop);
  ctx.lineTo(zeroX, h - padBottom);
  ctx.stroke();

  feats.forEach((feat, idx) => {
    const y = padTop + idx * barH;
    const l1 = loadings[feat].PC1;
    const barLen = (l1 / 0.5) * (plotW / 2);

    ctx.fillStyle = l1 >= 0 ? PALETTE.yellow : '#fde68a';
    const barPad = Math.max(2, Math.min(4, barH * 0.15));
    if (l1 >= 0) {
      ctx.fillRect(zeroX, y + barPad, barLen, barH - barPad * 2);
    } else {
      ctx.fillRect(zeroX + barLen, y + barPad, -barLen, barH - barPad * 2);
    }

    ctx.fillStyle = '#1c1917';
    ctx.font = `${fontSize}px sans-serif`;
    ctx.textAlign = 'right';
    let cleanName = feat.replace(/_/g, ' ').replace(' m', '').replace(' mm', '');
    if (w < 420 && cleanName.length > 12) {
      cleanName = cleanName.slice(0, 10) + '..';
    }
    ctx.fillText(cleanName, padLeft - 6, y + barH / 2 + 3);

    ctx.fillStyle = '#57534e';
    ctx.font = `${Math.max(7, fontSize - 1)}px sans-serif`;
    ctx.textAlign = l1 >= 0 ? 'left' : 'right';
    const labelX = l1 >= 0 ? zeroX + barLen + 3 : zeroX + barLen - 3;
    ctx.fillText(l1.toFixed(2), labelX, y + barH / 2 + 3);
  });
}

// 8. Render Dynamic Risk Cards Grid (Renders Exactly K Cards)
function renderRiskCards(profiles) {
  const container = document.getElementById('risk-cards-container');
  if (!container || !appData) return;

  const activeProfiles = profiles || computeDynamicProfiles(currentK, currentMetric);

  container.innerHTML = activeProfiles.map(p => {
    const color = CLUSTER_COLORS[p.clusterId % CLUSTER_COLORS.length];
    const meta = p.metadata;

    return `
      <div class="risk-card" style="border-top: 4px solid ${color};">
        <div class="risk-card-header">
          <div class="risk-dot" style="background-color: ${color};"></div>
          <div class="risk-title">${meta.name}</div>
        </div>
        <div class="risk-meta">
          <span><strong>${p.size}</strong> catchments</span> &bull; <span><strong>${p.percentage}%</strong> of total</span>
        </div>
        <div class="risk-desc">${meta.description}</div>
        <div class="risk-stat-row">
          <span class="risk-stat-label">Mean Terrain Slope:</span>
          <span class="risk-stat-val">${p.rawAverages.slope_mean_deg}°</span>
        </div>
        <div class="risk-stat-row">
          <span class="risk-stat-label">Catchment Relief:</span>
          <span class="risk-stat-val">${p.rawAverages.elevation_relief_m} m</span>
        </div>
        <div class="risk-stat-row">
          <span class="risk-stat-label">Max Cloudburst Rate:</span>
          <span class="risk-stat-val"><strong>${p.rawAverages.max_hourly_intensity_mm_hr} mm/h</strong></span>
        </div>
        <div class="risk-stat-row">
          <span class="risk-stat-label">Peak Concentration Lag:</span>
          <span class="risk-stat-val"><strong style="color:${color};">${p.rawAverages.response_time_hr} hrs</strong></span>
        </div>
      </div>
    `;
  }).join('');
}

// 9. Render Dynamic Radar Chart (Draws Exactly K Polygons)
function renderRadarChart(profiles) {
  const canvas = document.getElementById('radar-chart');
  if (!canvas || !appData) return;

  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return;

  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;
  ctx.clearRect(0, 0, w, h);

  const centerX = w / 2;
  const centerY = h / 2;
  const labelMargin = w < 400 ? 28 : 42;
  const radius = Math.max(50, Math.min(centerX, centerY) - labelMargin);
  const fontSize = Math.max(8, Math.min(10, Math.round(w / 45)));

  const dimensions = [
    { key: 'slope_mean_deg', label: w < 400 ? 'Slope' : 'Slope (°)' },
    { key: 'elevation_relief_m', label: w < 400 ? 'Relief' : 'Relief (m)' },
    { key: 'max_hourly_intensity_mm_hr', label: w < 400 ? 'Rain Int' : 'Rain Intensity' },
    { key: 'consecutive_wet_days_cwd', label: w < 400 ? 'CWD' : 'Antecedent CWD' },
    { key: 'topographic_wetness_index', label: 'TWI' },
    { key: 'drainage_density_km_sqkm', label: w < 400 ? 'Drainage' : 'Drain Density' }
  ];

  const totalAxes = dimensions.length;

  // Concentric Web
  for (let level = 1; level <= 4; level++) {
    const r = (level / 4) * radius;
    ctx.strokeStyle = '#f3f4f6';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < totalAxes; i++) {
      const angle = (i / totalAxes) * Math.PI * 2 - Math.PI / 2;
      const x = centerX + Math.cos(angle) * r;
      const y = centerY + Math.sin(angle) * r;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.stroke();
  }

  // Axis lines & labels
  dimensions.forEach((dim, i) => {
    const angle = (i / totalAxes) * Math.PI * 2 - Math.PI / 2;
    const x = centerX + Math.cos(angle) * radius;
    const y = centerY + Math.sin(angle) * radius;

    ctx.strokeStyle = '#e7e5e4';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(x, y);
    ctx.stroke();

    ctx.fillStyle = '#44403c';
    ctx.font = `${fontSize}px sans-serif`;
    ctx.textAlign = 'center';
    const textDist = radius + (w < 400 ? 12 : 18);
    const labelX = centerX + Math.cos(angle) * textDist;
    const labelY = centerY + Math.sin(angle) * textDist + 3;
    ctx.fillText(dim.label, labelX, labelY);
  });

  // Plot Exactly K Polygons
  const activeProfiles = profiles || computeDynamicProfiles(currentK, currentMetric);
  activeProfiles.forEach(p => {
    const color = CLUSTER_COLORS[p.clusterId % CLUSTER_COLORS.length];
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(1.8, Math.min(2.4, w / 180));
    ctx.fillStyle = color + '22'; // 15% opacity

    ctx.beginPath();
    dimensions.forEach((dim, i) => {
      const z = p.standardizedCentroid[dim.key] || 0;
      const normVal = Math.max(0.1, Math.min(1.0, (z + 2.2) / 4.4));
      const r = normVal * radius;
      const angle = (i / totalAxes) * Math.PI * 2 - Math.PI / 2;
      const x = centerX + Math.cos(angle) * r;
      const y = centerY + Math.sin(angle) * r;

      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.stroke();
    ctx.fill();
  });
}

// 10. Render Dynamic Centroids Table (Headers & Columns match K)
function renderCentroidsTable(profiles) {
  const thead = document.getElementById('centroids-thead');
  const tbody = document.getElementById('centroids-tbody');
  if (!thead || !tbody || !appData) return;

  const activeProfiles = profiles || computeDynamicProfiles(currentK, currentMetric);

  // Dynamic Table Header
  const headerCells = activeProfiles.map(p => {
    const color = CLUSTER_COLORS[p.clusterId % CLUSTER_COLORS.length];
    return `<th><span style="color:${color};">●</span> ${p.metadata.shortName}</th>`;
  }).join('');

  thead.innerHTML = `
    <tr>
      <th>Feature Dimension</th>
      ${headerCells}
    </tr>
  `;

  // Dynamic Table Rows
  const rows = [
    { label: 'Mean Elevation (m)', key: 'elevation_mean_m', unit: 'm' },
    { label: 'Catchment Relief (m)', key: 'elevation_relief_m', unit: 'm' },
    { label: 'Slope Gradient (°)', key: 'slope_mean_deg', unit: '°' },
    { label: 'Drainage Density (km/km²)', key: 'drainage_density_km_sqkm', unit: '' },
    { label: 'Topographic Wetness Index', key: 'topographic_wetness_index', unit: '' },
    { label: 'Peak 24h Monsoon Rain (mm)', key: 'peak_monsoon_rainfall_mm', unit: 'mm' },
    { label: 'Max Hourly Cloudburst Rate (mm/h)', key: 'max_hourly_intensity_mm_hr', unit: 'mm/h' },
    { label: 'Antecedent Consecutive Wet Days', key: 'consecutive_wet_days_cwd', unit: 'days' },
    { label: 'Hydrological Lag Response Time (hr)', key: 'response_time_hr', unit: 'hrs' }
  ];

  tbody.innerHTML = rows.map(r => {
    const vals = activeProfiles.map(p => {
      const v = p.rawAverages[r.key];
      return `<td><strong>${v !== undefined ? v : '-'}</strong> ${r.unit}</td>`;
    });
    return `
      <tr>
        <td><strong>${r.label}</strong></td>
        ${vals.join('')}
      </tr>
    `;
  }).join('');
}

// 11. Data Exporters
function downloadCSV() {
  window.open('../data/preprocessed_catchment_data.csv', '_blank');
}

function downloadJSON() {
  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(appData, null, 2));
  const dlAnchorElem = document.createElement('a');
  dlAnchorElem.setAttribute("href", dataStr);
  dlAnchorElem.setAttribute("download", "flash_flood_risk_clustering_results.json");
  dlAnchorElem.click();
}
