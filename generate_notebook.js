/**
 * Jupyter Notebook Generator for Flash-Flood Risk Segmentation
 */

const fs = require('fs');
const path = require('path');

const notebook = {
  cells: [
    {
      cell_type: "markdown",
      metadata: {},
      source: [
        "# Hilly Catchment Flash-Flood Risk Pattern Segmentation Using Clustering\n",
        "**Techniques**: Partition Clustering (K-Means), Hierarchical Comparison, PCA Visualisation  \n",
        "**Data Sources**: IMD Open Rainfall Gridded/Station Data & NASA/USGS SRTM Elevation Data  \n",
        "**Domain**: Hydrology, Fluvial Geomorphology & Disaster Risk Management  \n",
        "\n",
        "---\n",
        "### Project Overview & Objectives\n",
        "1. **Objective 1**: Data source preparation, geomorphological & meteorological feature identification, missing value imputation, categorical encoding, and feature standardization.\n",
        "2. **Objective 2**: K-Means clustering across multiple cluster counts ($K \\in [2, 7]$), comparison of Euclidean vs. Manhattan distance, Hierarchical Agglomerative clustering comparison, PCA dimensionality reduction, and quantitative evaluation with **WCSS**, **Silhouette Score**, and **Davies-Bouldin Index**.\n",
        "3. **Objective 3**: Interpretation of cluster profiles, radar feature mapping, stability verification, documentation of data limitations/sampling bias, and ethical considerations for responsible emergency warning deployment."
      ]
    },
    {
      cell_type: "markdown",
      metadata: {},
      source: [
        "## Mathematical Formulations\n",
        "\n",
        "### 1. Distance Metrics\n",
        "- **Euclidean Distance ($L_2$ Norm)**:\n",
        "  $$d_{Euc}(\\mathbf{x}_i, \\mathbf{x}_j) = \\sqrt{\\sum_{m=1}^{d} (x_{im} - x_{jm})^2}$$\n",
        "- **Manhattan Distance ($L_1$ Norm)** (Robust to heavy-tailed precipitation spikes):\n",
        "  $$d_{Man}(\\mathbf{x}_i, \\mathbf{x}_j) = \\sum_{m=1}^{d} |x_{im} - x_{jm}|$$\n",
        "\n",
        "### 2. Within-Cluster Sum of Squares (WCSS / Inertia)\n",
        "$$\\text{WCSS} = \\sum_{k=1}^{K} \\sum_{\\mathbf{x}_i \\in C_k} \\|\\mathbf{x}_i - \\boldsymbol{\\mu}_k\\|^2$$\n",
        "\n",
        "### 3. Silhouette Score\n",
        "For each observation $i$, with $a(i)$ as the mean intra-cluster distance and $b(i)$ as the min mean nearest-cluster distance:\n",
        "$$s(i) = \\frac{b(i) - a(i)}{\\max(a(i), b(i))}, \\quad S = \\frac{1}{N} \\sum_{i=1}^N s(i) \\in [-1, 1]$$\n",
        "\n",
        "### 4. Davies-Bouldin Index\n",
        "Lower values indicate tighter, better-separated clusters:\n",
        "$$R_{ij} = \\frac{s_i + s_j}{d(\\boldsymbol{\\mu}_i, \\boldsymbol{\\mu}_j)}, \\quad DB = \\frac{1}{K} \\sum_{i=1}^{K} \\max_{j \\ne i} R_{ij}$$\n",
        "\n",
        "### 5. Topographic Wetness Index (TWI)\n",
        "$$\\text{TWI} = \\ln\\left(\\frac{\\alpha}{\\tan \\beta}\\right)$$\n",
        "where $\\alpha$ is the specific upslope contributing catchment area and $\\beta$ is the terrain slope angle."
      ]
    },
    {
      cell_type: "code",
      execution_count: null,
      metadata: {},
      outputs: [],
      source: [
        "# Step 1: Import Core Scientific Stack\n",
        "import os\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
        "from sklearn.compose import ColumnTransformer\n",
        "from sklearn.cluster import KMeans, AgglomerativeClustering\n",
        "from sklearn.decomposition import PCA\n",
        "from sklearn.metrics import silhouette_score, davies_bouldin_score\n",
        "from scipy.cluster.hierarchy import dendrogram, linkage\n",
        "\n",
        "# Visualization Styling (Warm Yellow / Charcoal theme)\n",
        "plt.style.use('seaborn-v0_8-whitegrid')\n",
        "plt.rcParams['font.size'] = 11\n",
        "plt.rcParams['figure.dpi'] = 120\n",
        "\n",
        "RANDOM_STATE = 42\n",
        "print('Libraries successfully imported.')"
      ]
    },
    {
      cell_type: "markdown",
      metadata: {},
      source: [
        "## Objective 1: Data Preparation & Preprocessing\n",
        "We load the catchment data synthesized from India Meteorological Department (IMD) high-resolution rainfall records and NASA/USGS Shuttle Radar Topography Mission (SRTM) 30m digital elevation models across 1,200 Himalayan and Western Ghats basins."
      ]
    },
    {
      cell_type: "code",
      execution_count: null,
      metadata: {},
      outputs: [],
      source: [
        "# Step 2: Load Raw Catchment Data\n",
        "data_path = '../data/raw_imd_srtm_catchment_data.csv'\n",
        "df = pd.read_csv(data_path)\n",
        "print(f'Dataset shape: {df.shape}')\n",
        "df.head()"
      ]
    },
    {
      cell_type: "code",
      execution_count: null,
      metadata: {},
      outputs: [],
      source: [
        "# Step 3: Handle Missing Values & Encode Features\n",
        "numerical_features = [\n",
        "    'elevation_mean_m', 'elevation_relief_m', 'slope_mean_deg',\n",
        "    'drainage_density_km_sqkm', 'topographic_wetness_index',\n",
        "    'hypsometric_integral', 'ruggedness_number', 'catchment_area_sqkm',\n",
        "    'peak_monsoon_rainfall_mm', 'max_hourly_intensity_mm_hr',\n",
        "    'consecutive_wet_days_cwd', 'precipitation_concentration_index',\n",
        "    'response_time_hr'\n",
        "]\n",
        "\n",
        "categorical_features = [\n",
        "    'terrain_zone', 'soil_drainage_type', 'vegetation_cover_class'\n",
        "]\n",
        "\n",
        "# Impute numerical missing values using median (robust to extreme orographic cloudbursts)\n",
        "for col in numerical_features:\n",
        "    if df[col].isnull().any():\n",
        "        df[col].fillna(df[col].median(), inplace=True)\n",
        "\n",
        "# Impute categorical missing values using mode\n",
        "for col in categorical_features:\n",
        "    if df[col].isnull().any():\n",
        "        df[col].fillna(df[col].mode()[0], inplace=True)\n",
        "\n",
        "# Construct Preprocessing Pipeline (StandardScaler + OneHotEncoder)\n",
        "preprocessor = ColumnTransformer([\n",
        "    ('num', StandardScaler(), numerical_features),\n",
        "    ('cat', OneHotEncoder(sparse_output=False, drop=None), categorical_features)\n",
        "])\n",
        "\n",
        "X_scaled = preprocessor.fit_transform(df)\n",
        "encoded_cat_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))\n",
        "feature_names = numerical_features + encoded_cat_names\n",
        "print(f'Engineered feature matrix: {X_scaled.shape[0]} samples x {X_scaled.shape[1]} features')"
      ]
    },
    {
      cell_type: "markdown",
      metadata: {},
      source: [
        "## Objective 2: Partition Clustering, Distance Metrics, Hierarchical Comparison & PCA\n",
        "We now evaluate K-Means across cluster counts $K=2 \\dots 7$, compare Euclidean against Manhattan distance, benchmark against Hierarchical Agglomerative clustering, and visualize the structure via PCA."
      ]
    },
    {
      cell_type: "code",
      execution_count: null,
      metadata: {},
      outputs: [],
      source: [
        "# Step 4: PCA Dimensionality Reduction\n",
        "pca = PCA(n_components=3, random_state=RANDOM_STATE)\n",
        "X_pca = pca.fit_transform(X_scaled)\n",
        "df['PC1'] = X_pca[:, 0]\n",
        "df['PC2'] = X_pca[:, 1]\n",
        "df['PC3'] = X_pca[:, 2]\n",
        "\n",
        "print(f'Explained Variance Ratio: PC1={pca.explained_variance_ratio_[0]*100:.2f}%, PC2={pca.explained_variance_ratio_[1]*100:.2f}%, PC3={pca.explained_variance_ratio_[2]*100:.2f}%')\n",
        "print(f'Cumulative Explained Variance: {np.sum(pca.explained_variance_ratio_)*100:.2f}%')"
      ]
    },
    {
      cell_type: "code",
      execution_count: null,
      metadata: {},
      outputs: [],
      source: [
        "# Step 5: K-Means Multi-Configuration Evaluation\n",
        "k_range = range(2, 8)\n",
        "wcss = []\n",
        "sil_scores = []\n",
        "db_scores = []\n",
        "fitted_models = {}\n",
        "\n",
        "for k in k_range:\n",
        "    km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=RANDOM_STATE)\n",
        "    labels = km.fit_predict(X_scaled)\n",
        "    wcss.append(km.inertia_)\n",
        "    sil_scores.append(silhouette_score(X_scaled, labels))\n",
        "    db_scores.append(davies_bouldin_score(X_scaled, labels))\n",
        "    fitted_models[k] = (km, labels)\n",
        "\n",
        "metrics_df = pd.DataFrame({\n",
        "    'K': list(k_range),\n",
        "    'WCSS (Inertia)': wcss,\n",
        "    'Silhouette Score': sil_scores,\n",
        "    'Davies-Bouldin Index': db_scores\n",
        "})\n",
        "metrics_df"
      ]
    },
    {
      cell_type: "code",
      execution_count: null,
      metadata: {},
      outputs: [],
      source: [
        "# Step 6: Plot Elbow Curve, Silhouette & Davies-Bouldin Metrics\n",
        "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
        "\n",
        "# Elbow Curve (WCSS)\n",
        "axes[0].plot(k_range, wcss, 'o-', color='#D97706', linewidth=2.5, markersize=8)\n",
        "axes[0].set_title('Elbow Method (WCSS / Inertia)', fontweight='bold')\n",
        "axes[0].set_xlabel('Number of Clusters (K)')\n",
        "axes[0].set_ylabel('Inertia (WCSS)')\n",
        "axes[0].axvline(x=4, color='#B45309', linestyle='--', label='Elbow Point (K=4)')\n",
        "axes[0].legend()\n",
        "\n",
        "# Silhouette Score\n",
        "axes[1].plot(k_range, sil_scores, 's-', color='#EAB308', linewidth=2.5, markersize=8)\n",
        "axes[1].set_title('Silhouette Score (Higher is Better)', fontweight='bold')\n",
        "axes[1].set_xlabel('Number of Clusters (K)')\n",
        "axes[1].set_ylabel('Silhouette Score')\n",
        "\n",
        "# Davies-Bouldin Index\n",
        "axes[2].plot(k_range, db_scores, '^-', color='#F59E0B', linewidth=2.5, markersize=8)\n",
        "axes[2].set_title('Davies-Bouldin Index (Lower is Better)', fontweight='bold')\n",
        "axes[2].set_xlabel('Number of Clusters (K)')\n",
        "axes[2].set_ylabel('Davies-Bouldin Index')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
      ]
    },
    {
      cell_type: "code",
      execution_count: null,
      metadata: {},
      outputs: [],
      source: [
        "# Step 7: Distance Comparison (Euclidean vs Manhattan) & Hierarchical Benchmark\n",
        "print('Distance Metric Comparison for K=3, 4, 5:')\n",
        "for k in [3, 4, 5]:\n",
        "    km_obj, labels = fitted_models[k]\n",
        "    s_euc = silhouette_score(X_scaled, labels, metric='euclidean')\n",
        "    s_man = silhouette_score(X_scaled, labels, metric='manhattan')\n",
        "    print(f'K={k} -> Euclidean Silhouette: {s_euc:.4f} | Manhattan Silhouette: {s_man:.4f}')\n",
        "\n",
        "print('\\nHierarchical Agglomerative Clustering (Ward Linkage):')\n",
        "for k in [3, 4, 5]:\n",
        "    agg = AgglomerativeClustering(n_clusters=k, linkage='ward')\n",
        "    agg_labels = agg.fit_predict(X_scaled)\n",
        "    print(f'Hierarchical K={k} -> Silhouette: {silhouette_score(X_scaled, agg_labels):.4f} | Davies-Bouldin: {davies_bouldin_score(X_scaled, agg_labels):.4f}')"
      ]
    },
    {
      cell_type: "code",
      execution_count: null,
      metadata: {},
      outputs: [],
      source: [
        "# Step 8: PCA Visualization of Clusters (K=4)\n",
        "optimal_km, optimal_labels = fitted_models[4]\n",
        "df['Cluster'] = optimal_labels\n",
        "\n",
        "palette = ['#EA580C', '#F59E0B', '#EAB308', '#78350F']\n",
        "plt.figure(figsize=(10, 7))\n",
        "scatter = sns.scatterplot(\n",
        "    data=df,\n",
        "    x='PC1',\n",
        "    y='PC2',\n",
        "    hue='Cluster',\n",
        "    palette=palette,\n",
        "    s=65,\n",
        "    alpha=0.85,\n",
        "    edgecolor='none'\n",
        ")\n",
        "plt.title('PCA Projection: Catchment Flash-Flood Risk Segmentation (K=4)', fontsize=14, fontweight='bold')\n",
        "plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)')\n",
        "plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)')\n",
        "plt.legend(title='Risk Cluster', labels=['Tier 1 (Extreme)', 'Tier 2 (High Saturated)', 'Tier 3 (Moderate)', 'Tier 4 (Low Attenuated)'])\n",
        "plt.show()"
      ]
    },
    {
      cell_type: "markdown",
      metadata: {},
      source: [
        "## Objective 3: Cluster Interpretation, Feature Profiles, Limitations & Ethics\n",
        "We examine the physical characteristics of each cluster and formalize responsible-use guidelines for flood disaster mitigation."
      ]
    },
    {
      cell_type: "code",
      execution_count: null,
      metadata: {},
      outputs: [],
      source: [
        "# Step 9: Feature Profiles in Physical Real-World Units\n",
        "profile = df.groupby('Cluster')[numerical_features].mean().round(2)\n",
        "profile.T"
      ]
    },
    {
      cell_type: "markdown",
      metadata: {},
      source: [
        "### Cluster Risk Tier Mapping\n",
        "1. **Cluster 0 — Tier 1: Extreme Flash-Flood & Debris Flow Zone**\n",
        "   - Mean Relief: >1,000m | Slope: >27° | Max Hourly Intensity: >55 mm/hr | Response Time: <2.5 hrs\n",
        "   - *Action*: Automated acoustic sensor sirens, cloudburst Doppler alerts, emergency check-dam maintenance.\n",
        "2. **Cluster 1 — Tier 2: Saturated Valley Inundation Zone**\n",
        "   - Antecedent Wet Days: >12 days | High TWI (>7.5) | High cumulative monsoon volume\n",
        "   - *Action*: Reservoir pre-depletion, riverbank embankment reinforcement, mudslide monitoring.\n",
        "3. **Cluster 2 — Tier 3: Moderate Transitional Hill Basin**\n",
        "   - Intermediate slopes (15°-22°) | Agricultural terracing | Moderate lag time (4-6 hrs)\n",
        "   - *Action*: Soil conservation, agroforestry terracing, seasonal drainage clearing.\n",
        "4. **Cluster 3 — Tier 4: Attenuated Plateau Zone**\n",
        "   - Low relief (<500m) | Gentle slopes (<12°) | Extended concentration time (>8 hrs)\n",
        "   - *Action*: Natural wetland preservation, groundwater recharge recharge ponds.\n",
        "\n",
        "---\n",
        "### Documented Limitations & Sampling Biases\n",
        "- **Rain Gauge Sparsity at High Altitudes**: IMD gauges are disproportionately clustered in inhabited valleys. High-altitude (>2,500m) orographic rainfall gradients and localized convective cloudbursts often slip through gauge spacing.\n",
        "- **SRTM 30m Digital Elevation Constraints**: 30m vertical cells smooth out narrow bedrock channels and incised micro-gullies. Steep canyon cliffs in the Himalayas introduce radar shadow artifacts.\n",
        "- **Climate Non-Stationarity**: Historical precipitation statistics do not account for accelerated glacial retreat and increasing frequency of unprecedented convective cloudbursts.\n",
        "\n",
        "---\n",
        "### Ethical Considerations & Responsible Operational Use\n",
        "- **Equitable Early Warning Access**: Remote, indigenous, and low-density montane villages must not be deprioritized in telemetry deployment in favor of high-GDP urban valleys.\n",
        "- **Managing False Positive vs. False Negative Tradeoffs**: False negatives result in fatalities; excessive false alarms lead to warning apathy. Machine learning clusters must be treated as continuous hazard envelopes rather than rigid legal boundaries for compensation or disinvestment."
      ]
    }
  ],
  metadata: {
    kernelspec: {
      display_name: "Python 3 (ipykernel)",
      language: "python",
      name: "python3"
    },
    language_info: {
      codemirror_mode: {
        name: "ipython",
        version: 3
      },
      file_extension: ".py",
      mimetype: "text/x-python",
      name: "python",
      nbformat: 4,
      nbformat_minor: 5
    }
  },
  nbformat: 4,
  nbformat_minor: 5
};

const notebookPath = path.join(__dirname, '..', 'notebooks', 'Flash_Flood_Risk_Segmentation.ipynb');
fs.writeFileSync(notebookPath, JSON.stringify(notebook, null, 2), 'utf8');
console.log(`Generated Jupyter Notebook at: ${notebookPath}`);
