# 🔍 Streamflow Analysis Using Machine Learning and Global Datasets (ERA5 + GloFAS)

This repository contains the code, methodology, and scripts required for hydrological analysis of streamflow using machine learning techniques. The approach combines reanalysis climate data (ERA5/ERA5-Land) and streamflow data from the CEMS GloFAS v4.0 global forecasting system. The main goal is to reconstruct and model discharge time series in poorly gauged regions, enabling robust water resource assessment.

---

## 📦 Repository Structure

### 📁 Required Data (Manual Download)

Due to large file sizes (over 100 MB), the contents of the `data/` folder are not included in the repository. You must download the corresponding datasets from the release tag or external sources and place them in the following structure:

- `data/` — Raw and preprocessed data files  
  - `climate/` — ERA5-Land reanalysis climate data  
  - `terrain/` — Flow direction and accumulation layers (e.g., from HydroSHEDS)  
  - `discharge/` — Streamflow data from GloFAS and extraction geometries (points or basins)  

- `notebooks/` — Jupyter notebooks for exploratory analysis and modeling  
  - `Extract_Basins.ipynb` — Delineation of watersheds using flow direction and accumulation rasters  
  - `Create_Regression_Models.ipynb` — Training and validation of machine learning models to predict streamflow from climate variables

- `src/` — Modular source code  
  - `SWIM.py` — Functions for climate-discharge modeling: preprocessing, model training, and prediction

- `requirements.txt` — Project dependencies

- `README.md` — This file

---

## 🌍 Data Sources

### 1. ERA5-Land (ECMWF - Copernicus)

- **Variables used:** precipitation, air temperature, potential evaporation, etc.  
- **Resolution:** 0.1° (~9 km), daily  
- **Access:** via [CDS API](https://cds.climate.copernicus.eu/)

### 2. GloFAS v4.0 (CEMS)

- **Variable:** daily simulated streamflow (LISFLOOD model)  
- **Period:** 1979 to present  
- **Access:** via [EWDS Portal](https://ewds.climate.copernicus.eu/datasets/cems-glofas-historical?tab=overview)  

---

## 🧠 Modeling Approach

The modeling workflow follows a structured pipeline to estimate river discharge using machine learning techniques driven by global climate reanalysis data:

1. **Climate and Discharge Data Preprocessing**  
   - Monthly aggregation of ERA5-Land variables (e.g., precipitation, temperature, radiation, wind)  
   - Extraction and temporal alignment of streamflow series from GloFAS v4.0  
   - Temporal normalization: shifting monthly dates to the first day and removing anomalies  
   - Spatial operations: clipping to basin geometry or selecting nearest pixel to pour point

2. **Model Training and Evaluation**  
   - Construction of feature matrices from gridded climate data  
   - Optional dimensionality reduction using PCA (configurable variance threshold)  
   - Training regression models: e.g., Support Vector Regression (SVR), Random Forest, XGBoost  
   - Evaluation using hydrological metrics:  
     - NSE (Nash–Sutcliffe Efficiency)  
     - R² (Coefficient of Determination)  
     - PBIAS (Percent Bias)  
     - RMSE (Root Mean Square Error)  
   - Automated selection and saving of the best-performing model

3. **Discharge Prediction from New Climate Data**  
   - Application of trained models on new or future climate datasets  
   - Output: simulated monthly streamflow series with performance visualization  
   - Comparison with observed series using scatter plots, time series, and cumulative flow curves  
   - Model and results exported to disk for reproducibility

Each of these steps is encapsulated in modular Python functions within the `SWIM.py` module, ensuring flexibility and reusability.

---

## 📚 References

- Joint Research Center, Copernicus Emergency Management Service (2019): River discharge and related historical data from the Global Flood Awareness System. Early Warning Data Store (EWDS). DOI: 10.24381/cds.a4fdd6b9 (Accessed on 01-JUN-2025)
- Muñoz Sabater, J. (2019): ERA5-Land monthly averaged data from 1950 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). DOI: 10.24381/cds.68d2bb30 (Accessed on 01-JUN-2025)

---

## ⚙️ Requirements

To reproduce the environment and run the code, follow the steps below.

### 📥 1. Clone the repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/<your-org-or-username>/SWIM_WaterBalanceModule.git
cd SWIM_WaterBalanceModule
```

### 🔧2.  Create the environment

```bash
conda env create -f environment.yml
conda activate SWIM_WaterBalance
pip install git+https://github.com/navass11/pysheds
```

---

## 👥 Authors

This project is developed by the Hydrology and Climate group at [IHCantabria](https://www.ihcantabria.com/)

---


## 📄 License

This project is licensed under the **GNU General Public License v3.0**.  
You are free to use, modify, and distribute this software under the terms of the GPL license.  
See the [`LICENSE`](./LICENSE) file for full legal terms.

> 🔗 More about the GPL-3.0 License: [https://www.gnu.org/licenses/gpl-3.0.html](https://www.gnu.org/licenses/gpl-3.0.html)