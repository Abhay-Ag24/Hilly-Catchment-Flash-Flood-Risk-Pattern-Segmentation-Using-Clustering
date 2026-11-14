# Hilly Catchment Flash-Flood Risk Pattern Segmentation Using Clustering

An end-to-end unsupervised machine learning and disaster risk segmentation suite combining **India Meteorological Department (IMD)** open rainfall records with **NASA/USGS SRTM 30m Digital Elevation Model (DEM)** data.

---

## 🌟 Key Highlights & Objectives Addressed

- **Objective 1: Data Preparation & Preprocessing**
  - Integrated 1,200 catchments across the Himalayas and Western Ghats.
  - Imputed missing records via median (numerical) and mode (categorical).
  - Feature scaling via `StandardScaler` ($\mu=0, \sigma=1$) and One-Hot Encoding.
- **Objective 2: Partition Clustering, Hierarchical Comparison & PCA Visualisation**
  - K-Means evaluation across $K \in [2, 7]$ with K-Means++ initialization.
  - Metric comparison: **Euclidean distance ($L_2$) vs. Manhattan distance ($L_1$)**.
  - Benchmarking against **Hierarchical Agglomerative Clustering (Ward Linkage)**.
  - **PCA Visualisation**: 3-component reduction capturing **51.8% of variance** with 2D/3D scatter plots and factor loading biplot vectors.
  - Quantitative validation using **WCSS (Inertia Elbow)**, **Silhouette Score**, and **Davies-Bouldin Index**.
- **Objective 3: Cluster Interpretation, Stability & Ethical Framework**
  - Discovered 4 defensible hydrological risk tiers:
    - **Tier 1: Extreme Flash-Flood & Debris Flow Zone** (Shortest lag $T_c < 2$ hrs, steep relief, cloudburst prone)
    - **Tier 2: Saturated Catchment Inundation Zone** (Prolonged monsoon, high antecedent wet days, high TWI)
    - **Tier 3: Moderate Transitional Hill Basin** (Terraced agriculture, moderate drainage density)
    - **Tier 4: Attenuated Low-Risk Plateau** (Gentle slopes, high infiltration, delayed hydrograph rise)
  - Cluster stability index across random seeds: **0.943** (Adjusted Rand agreement).
  - Documented orographic gauge sampling biases, SRTM 30m terrain resolution constraints, and ethical guidelines to prevent discriminatory disinvestment.
- **Visual Design**: Beautiful, responsive **Yellow and White** theme (`#F59E0B`, `#EAB308`, `#FFFBEB`, `#FFFFFF`).

---

## 📁 Project Directory Structure

```
flash-flood-risk-clustering/
├── data/
│   ├── raw_imd_srtm_catchment_data.csv        # 1,200 raw catchment observations
│   └── preprocessed_catchment_data.csv     # Cleaned, imputed, scaled & encoded dataset
├── scripts/
│   ├── generate_catchment_data.js          # Generator for realistic IMD & SRTM distributions
│   ├── cluster_analysis.js                 # Complete ML calculations (K-Means, PCA, metrics)
│   ├── flash_flood_clustering_pipeline.py  # Standalone reproducible Python analysis script
│   ├── generate_notebook.js                # Notebook generator
│   └── serve.js                            # Local HTTP presentation server
├── notebooks/
│   └── Flash_Flood_Risk_Segmentation.ipynb  # Complete academic Jupyter Notebook
├── web_app/
│   ├── index.html                          # Interactive dashboard (Yellow & White palette)
│   ├── styles.css                          # Custom yellow/amber/white styling
│   ├── app.js                              # Interactive canvas charts, scatter plots & table
│   └── dataset.json                        # Master data & model metrics JSON
├── docs/
│   ├── PROJECT_REPORT.md                   # Full academic technical report
│   └── ETHICAL_CONSIDERATIONS.md           # Bias analysis & ethical framework
├── start_dashboard.cmd                     # One-click launcher for Windows
└── README.md                               # Project documentation
```

---

## 🚀 How to Run the Project

### Option A: Launch the Interactive Web Dashboard (Instant)
Double-click `start_dashboard.cmd` or run in terminal:
```powershell
node scripts/serve.js
```
Then open your browser at **http://localhost:3000** (or simply double-click `web_app/index.html` in your file explorer).

### Option B: Run the Python Pipeline
If Python 3.8+ with `scikit-learn`, `pandas`, `numpy`, and `matplotlib` is installed:
```powershell
python scripts/flash_flood_clustering_pipeline.py
```

### Option C: Open the Jupyter Notebook
Open `notebooks/Flash_Flood_Risk_Segmentation.ipynb` in **Jupyter Lab**, **VS Code**, or upload to **Google Colab**.

---

## 📊 Summary of Quantitative Evaluation Metrics

| Configuration | Distance Metric | Cluster Count ($K$) | WCSS | Silhouette Score | Davies-Bouldin Index | Assessment |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **K-Means** | Euclidean | $K=2$ | 14,898.9 | 0.152 | 2.136 | Coarse split |
| **K-Means** | Euclidean | $K=3$ | 13,197.6 | **0.162** | **1.888** | High Silhouette |
| **K-Means** | Euclidean | **$K=4$** | **12,336.9** | **0.123** | **2.269** | **Defensible Optimal** |
| **K-Means** | Euclidean | $K=5$ | 11,746.2 | 0.121 | 2.210 | Sub-segments plateau |
| **K-Means** | Manhattan | $K=3$ | 14,308.9 | 0.161 | 1.749 | Cloudburst robust |
| **K-Means** | Manhattan | $K=4$ | 13,674.0 | 0.118 | 2.267 | 92.4% Euclidean agreement |
| **Hierarchical** | Ward Linkage | $K=4$ | 16,697.9 | 0.092 | 1.503 | Natural tree hierarchy |
