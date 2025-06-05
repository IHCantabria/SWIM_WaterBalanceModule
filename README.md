# 🔍 Streamflow Analysis Using Machine Learning and Global Datasets (ERA5 + GloFAS)

This repository contains the code, methodology, and scripts required for hydrological analysis of streamflow using machine learning techniques. The approach combines reanalysis climate data (ERA5/ERA5-Land) and streamflow data from the CEMS GloFAS v4.0 global forecasting system. The main goal is to reconstruct and model discharge time series in poorly gauged regions, enabling robust water resource assessment.

---

## 📦 Repository Structure

- `data/` — Raw and preprocessed data files  
  - `climate/` — ERA5-Land downloads  
  - `terrain/` — Historical GloFAS streamflow files  
  - `dicharge/` — Geometries for extraction and aggregation  

- `notebooks/` — Jupyter notebooks for exploratory analysis and modeling

- `src/` — Modular source code  
  - `preprocessing.py` — Cleaning, interpolation and aggregation  
  - `extraction.py` — NetCDF data extraction by coordinates or polygons  
  - `models.py` — ML model definition, training and validation  
  - `utils.py` — Auxiliary functions  

- `outputs/` — Generated results (models, figures, reports)

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

The modeling workflow includes the following steps:

1. **Preprocessing climate and streamflow data**  
   - Spatial clipping and aggregation by basin or point  
   - Gap filling and normalization

2. **Training machine learning models**  
   - Algorithms: Random Forest, Gradient Boosting, LSTM  
   - Cross-validation by time period and region  
   - Metrics: NSE, R², RMSE, PBIAS

3. **Historical reconstruction and scenario simulation**  
   - Use of synthetic or reanalysis-based forcing  
   - Models trained with ERA5 + GloFAS series

---

## ⚙️ Requirements

Install dependencies using:

```bash
pip install -r requirements.txt