import os
import geopandas as gpd
import fiona
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pysheds.grid import Grid
import matplotlib.gridspec as gridspec
import rioxarray
import xarray as xr
import pandas as pd
import numpy as np

import joblib
from sklearn.preprocessing import StandardScaler
import hydroeval as he
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer
from sklearn.exceptions import ConvergenceWarning

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

def nse_func(X_train_scaled, Y_train_scaled):
    nse  = he.evaluator(he.nse, X_train_scaled, Y_train_scaled)[0]
    return nse

def bias_func(X_train_scaled, Y_train_scaled):
    bias  = np.abs(he.evaluator(he.pbias, X_train_scaled, Y_train_scaled)[0])
    return bias
    
def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value**2

class WaterBalanceModule:
    def __init__(self):
        pass

    def delineate_watershed_from_raster(
        self,
        dir_raster_path,
        acc_raster_path,
        output_shapefile_path,
        pour_point_coords,
        acc_threshold=1000,
        target_epsg=4326
    ):
        if not os.path.exists(dir_raster_path):
            raise FileNotFoundError(f"Flow direction raster not found: {dir_raster_path}")
        if not os.path.exists(acc_raster_path):
            raise FileNotFoundError(f"Accumulation raster not found: {acc_raster_path}")

        grid = Grid.from_raster(dir_raster_path)
        fdir = grid.read_raster(dir_raster_path)
        acc = grid.read_raster(acc_raster_path)

        dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

        x_snap, y_snap = grid.snap_to_mask(acc > acc_threshold, pour_point_coords)
        catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap, xytype='coordinate')
        grid.clip_to(catch)

        shapes = grid.polygonize()
        schema = {
            'geometry': 'Polygon',
            'properties': {'LABEL': 'float:16'}
        }

        with fiona.open(output_shapefile_path, 'w',
                        driver='ESRI Shapefile',
                        crs=grid.crs.srs,
                        schema=schema) as c:
            for i, (shape, value) in enumerate(shapes):
                rec = {
                    'geometry': shape,
                    'properties': {'LABEL': str(value)},
                    'id': str(i)
                }
                c.write(rec)

        gdf = gpd.read_file(output_shapefile_path)
        gdf = gdf.to_crs(epsg=target_epsg)

        return gdf, (x_snap, y_snap)

    def plot_catchment_country_and_zoom(self, gdf, pour_point_coords, title="Catchment Map",
                                     country_shapefile_path='../data/terrain/ne_110m_admin_0_countries.shp',
                                     output_path=None):
        """
        Plot a two-panel map (country + zoom), auto-detect country by intersection, and optionally save to file.

        Parameters:
            gdf (GeoDataFrame): Catchment polygon (EPSG:4326)
            pour_point_coords (tuple): (lon, lat)
            title (str): Title of the figure
            country_shapefile_path (str): Path to Natural Earth shapefile of countries
            output_path (str or None): If set, export PNG to this path
        """
        # 1. Load country shapefile and ensure CRS
        world = gpd.read_file(country_shapefile_path).to_crs("EPSG:4326")
        gdf = gdf.to_crs("EPSG:4326")

        # 2. Find country or countries that intersect the catchment
        matching_countries = world[world.intersects(gdf.unary_union)]

        if matching_countries.empty:
            raise ValueError("No country intersects the watershed geometry.")

        # 3. Select first matching country (or handle multiple later)
        country_row = matching_countries.iloc[0]
        country_name = country_row.get('ADMIN', country_row.get('name', 'Unknown'))
        country_geom = country_row.geometry
        country_bounds = country_geom.bounds

        # 4. Extract pour point and catchment bounds
        lon, lat = pour_point_coords
        catch_bounds = gdf.total_bounds
        dx = catch_bounds[2] - catch_bounds[0]
        dy = catch_bounds[3] - catch_bounds[1]

        # 5. Create figure
        fig = plt.figure(figsize=(10, 5))
        #fig.suptitle(f"{title}\n({country_name})", fontsize=12)

        gs = gridspec.GridSpec(1, 2, width_ratios=[2.2, 1.3])

        # --------- Panel 1: Country View ---------
        ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        ax1.set_title(f"Location in {country_name}")
        ax1.set_extent([
            country_bounds[0] - 1,
            country_bounds[2] + 1,
            country_bounds[1] - 1,
            country_bounds[3] + 1
        ])

        ax1.add_feature(cfeature.LAND, facecolor='lightgray')
        ax1.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax1.add_feature(cfeature.COASTLINE)
        ax1.add_feature(cfeature.BORDERS, linestyle=':')
        ax1.add_feature(cfeature.RIVERS)
        gdf.plot(ax=ax1, facecolor='none', edgecolor='blue', linewidth=1.5, zorder=3)
        ax1.plot(lon, lat, marker='o', color='red', transform=ccrs.PlateCarree())

        gl1 = ax1.gridlines(draw_labels=True, x_inline=False, y_inline=False)
        gl1.top_labels = gl1.right_labels = False

        # --------- Panel 2: Zoom to Catchment ---------
        ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
        ax2.set_title("Catchment Zoom")
        ax2.set_extent([
            catch_bounds[0] - dx * 0.1,
            catch_bounds[2] + dx * 0.1,
            catch_bounds[1] - dy * 0.1,
            catch_bounds[3] + dy * 0.1
        ])

        ax2.add_feature(cfeature.LAND)
        ax2.add_feature(cfeature.COASTLINE)
        ax2.add_feature(cfeature.BORDERS, linestyle=':')
        ax2.add_feature(cfeature.RIVERS)
        gdf.boundary.plot(ax=ax2, edgecolor='blue', linewidth=2, zorder=3)
        gdf.plot(ax=ax2, facecolor='none', edgecolor='blue', alpha=0.3, zorder=2)
        ax2.plot(lon, lat, marker='o', color='red', markersize=8, transform=ccrs.PlateCarree(), label="Pour Point")

        gl2 = ax2.gridlines(draw_labels=True, x_inline=False, y_inline=False)
        gl2.top_labels = gl2.right_labels = False
        ax2.legend(loc="lower right")

        plt.tight_layout()
        plt.subplots_adjust(top=0.88)

        # --------- Save or show figure ---------
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Map saved to: {output_path}")
        else:
            plt.show()


    def extract_monthly_discharge_from_netcdf(self, nc_path, points):
        """
        Extract and resample monthly discharge at given coordinate points from a NetCDF file.

        Parameters:
            nc_path (str): Path to NetCDF file.
            points (list of tuples): List of (lon, lat) points to extract, e.g., [(x1, y1), (x2, y2)].

        Returns:
            dict: Dictionary with point index as key and monthly discharge (xarray.DataArray) as value.
        """
        import xarray as xr
        import pandas as pd

        ds = xr.open_dataset(nc_path)

        varname='dis24'
        time_dim='valid_time'

        if varname not in ds:
            raise ValueError(f"Variable '{varname}' not found in NetCDF.")

        monthly_discharge = {}

        for i, (x, y) in enumerate(points, start=1):
            discharge = ds[varname].sel(latitude=y, longitude=x, method='nearest')
            monthly = discharge.resample({time_dim: 'ME'}).mean()
            monthly[time_dim] = pd.to_datetime(monthly[time_dim].data) - pd.offsets.MonthEnd(0) + pd.offsets.Day(1)
            monthly[time_dim] = pd.to_datetime(monthly[time_dim].values).normalize()
            monthly_discharge[f'discharge_{i}'] = monthly

        return monthly_discharge
    

    def extract_and_plot_climate_variables(self,
                                       nc_path,
                                       basin_shapefile,
                                       time_range=('1979-01-01', '2025-07-01'),
                                       var_mapping=None):
        """
        Extract and plot mean climate variables clipped to a basin from a NetCDF file.

        Parameters:
            nc_path (str): Path to the NetCDF file containing climate data.
            basin_shapefile (str): Path to the shapefile defining the basin polygon.
            time_range (tuple): Start and end dates for temporal filtering (YYYY-MM-DD).
            var_mapping (dict): Dictionary mapping variable names in the NetCDF to human-readable titles,
                                e.g. {'t2m': 'Temperature', 'tp': 'Precipitation'}.

        Returns:
            xarray.Dataset: Dataset with clipped and filtered variables.
        """
        # 1. Open the NetCDF dataset
        ds = xr.open_dataset(nc_path)

        # 2. Normalize and shift time coordinates to first day of month
        ds['time'] = pd.to_datetime(ds['time'].data) - pd.offsets.MonthEnd(0) + pd.offsets.Day(1)
        ds['time'] = pd.to_datetime(ds['time'].values).normalize()

        # 3. Select time slice
        ds = ds.sel(time=slice(*time_range))

        # 4. Assign CRS for spatial operations
        ds = ds.rio.write_crs("EPSG:4326", inplace=False)

        # 5. Read basin shapefile and reproject to EPSG:4326
        basin = gpd.read_file(basin_shapefile).to_crs("EPSG:4326")

        # 6. Clip dataset using the basin geometry
        ds_clipped = ds.rio.clip(basin.geometry, basin.crs, all_touched=True, drop=True)

        # 7. Set default variable mappings if none provided
        if var_mapping is None:
            var_mapping = {
                't2m': 'Temperature',
                'ssrd': 'Radiation',
                'tp': 'Precipitation',
                'sfcWind': 'Wind'
            }

        color_maps = ["viridis", "plasma", "Blues", "cividis"]
        units = ["K", "J/m²", "m", "m/s"]

        # 8. Create 2×2 subplot figure for visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        axes = axes.flatten()

        # 9. Loop through each variable to compute temporal mean and plot
        for i, (var, title) in enumerate(var_mapping.items()):
            if var not in ds_clipped:
                print(f"Variable {var} not found in dataset, skipping.")
                continue

            var_mean = ds_clipped[var].mean(dim='time')
            ax = axes[i]

            # Plot spatial distribution
            c = ax.pcolormesh(var_mean.longitude, var_mean.latitude, var_mean,
                            cmap=color_maps[i], shading='auto', transform=ccrs.PlateCarree())

            ax.set_title(title, fontsize=14)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

            # Add colorbar
            cbar = fig.colorbar(c, ax=ax, orientation='vertical')
            cbar.set_label(units[i], fontsize=12)

            # Overlay basin boundary
            basin.boundary.plot(ax=ax, edgecolor='black', linewidth=1, transform=ccrs.PlateCarree())

        # 10. Final layout and rendering
        plt.tight_layout()
        plt.show()

        return ds_clipped
    
    def build_flow_models_from_climate_data(self,
                                        climate_dataset,
                                        discharge_series,
                                        output_dir='./Modelos1',
                                        predictors=['tp', 't2m', 'ssrd', 'sfcWind'],
                                        apply_pca=True,
                                        pca_var_threshold=0.95,
                                        test_size=0.2,
                                            random_state=42):
        """
        Build, train, evaluate and save multiple regression models to predict discharge from climate variables.

        Parameters:
            climate_dataset (xr.Dataset): Dataset with climate variables (monthly, clipped to basin).
            discharge_series (xr.DataArray or pd.Series): Observed discharge series aligned with time.
            output_dir (str): Directory to save models and scalers.
            predictors (list): Variable names in the dataset to use as predictors.
            apply_pca (bool): Whether to apply PCA to reduce dimensionality.
            pca_var_threshold (float): Explained variance threshold for PCA (if used).
            test_size (float): Proportion of data to reserve for testing.
            random_state (int): Random seed for reproducibility.
        """      

        os.makedirs(output_dir, exist_ok=True)

        # 1. Flatten climate variables to [time, gridpoints] and concatenate
        lat = climate_dataset.sizes['latitude']
        lon = climate_dataset.sizes['longitude']
        n_cells = lat * lon

        X = []
        for var in predictors:
            arr = climate_dataset[var].data.reshape(-1, n_cells)
            X.append(arr)
        X = np.hstack(X).astype(float)

        # 2. Remove columns with NaN in any predictor
        valid_mask = ~np.isnan(X).any(axis=0)
        X = X[:, valid_mask]

        # 3. Normalize and optionally apply PCA
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)

        if apply_pca:
            pca = PCA(n_components=pca_var_threshold)
            X_transformed = pca.fit_transform(X_scaled)
            joblib.dump(pca, os.path.join(output_dir, f'pipeline_Flow.pkl'))
        else:
            X_transformed = X_scaled

        joblib.dump(scaler_X, os.path.join(output_dir, f'scaler_Flow.pkl'))

        # 4. Prepare target
        if isinstance(discharge_series, xr.DataArray):
            y = discharge_series.values.reshape(-1, 1).astype(float)
            time = discharge_series['valid_time'].values
        else:
            y = discharge_series.values.reshape(-1, 1).astype(float)
            time = discharge_series.index

        # 5. Align and merge
        df_X = pd.DataFrame(X_transformed, index=climate_dataset['time'].values[:X_transformed.shape[0]])
        df_y = pd.DataFrame(y, index=discharge_series['valid_time'].values[:y.shape[0]], columns=['Flow'])
        common_index = df_X.index.intersection(df_y.index)
        df_X = df_X.loc[common_index]
        df_y = df_y.loc[common_index]
        data = pd.concat([df_y, df_X], axis=1).dropna()

        X_all = data.iloc[:, 1:]
        y_all = data.iloc[:, 0]

        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=test_size, random_state=random_state)

        # 6. Define models and hyperparameter grids
        scoring = {
            "r2": "r2",
            "nse": make_scorer(nse_func, greater_is_better=True),
            "bias": make_scorer(bias_func, greater_is_better=False),
            "explained_variance": "explained_variance",
            "neg_mean_squared_error": "neg_mean_squared_error"
        }

        param_grids = {
            'SVR': {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']},
            'LinearRegression': {},
            'DecisionTreeRegressor': {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]},
            'KNeighborsRegressor': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
            'ElasticNet': {'alpha': [0.1, 1], 'l1_ratio': [0.1, 0.5]},
            'GradientBoostingRegressor': {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.01], 'max_depth': [3, 5]},
            'AdaBoostRegressor': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'estimator__max_depth': [3, 5],
                'estimator__min_samples_split': [10],
                'estimator__min_samples_leaf': [1]
            },
            'MLPRegressor': {
                "hidden_layer_sizes": [(50, 50), (90, 30, 10)],
                "activation": ["relu"],
                "solver": ["adam"],
                'learning_rate': ['adaptive'],
                "alpha": [0.0001, 0.001],
                'max_iter': [1000, 2000, 3000]
            }
        }

        models = {
            'LinearRegression': LinearRegression(),
            'SVR': SVR(),
            'DecisionTreeRegressor': DecisionTreeRegressor(),
            'KNeighborsRegressor': KNeighborsRegressor(),
            'ElasticNet': ElasticNet(),
            'GradientBoostingRegressor': GradientBoostingRegressor(random_state=random_state),
            'AdaBoostRegressor': AdaBoostRegressor(estimator=DecisionTreeRegressor(random_state=random_state), random_state=random_state),
            'MLPRegressor': MLPRegressor(random_state=random_state)
        }

        best_models = {}
        metrics = pd.DataFrame(columns=['Model', 'R2', 'NSE', 'PBIAS'])

        for name, model in models.items():
            print(f"🔍 Training {name}...")
            grid = GridSearchCV(model, param_grids.get(name, {}), scoring=scoring, refit='r2', cv=5, verbose=0, n_jobs=-1)
            grid.fit(X_train, y_train)
            best_models[name] = grid.best_estimator_

            # Save model
            path_model = os.path.join(output_dir, f"{name}_best_model_Flow.joblib")
            joblib.dump(grid.best_estimator_, path_model)

            # Evaluate
            y_pred = grid.best_estimator_.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            nse = he.evaluator(he.nse, y_test.values, y_pred)[0]
            pbias = he.evaluator(he.pbias, y_test.values, y_pred)[0]

            metrics = pd.concat([metrics, pd.DataFrame([{
                'Model': name,
                'R2': r2,
                'NSE': nse,
                'PBIAS': pbias
            }])], ignore_index=True)

        metrics[['R2', 'NSE', 'PBIAS']] = metrics[['R2', 'NSE', 'PBIAS']].astype(float)
        metrics['PBIAS_abs'] = np.abs(metrics['PBIAS'])
        metrics['SCORE'] = metrics['R2'] + metrics['NSE'] - 0.01 * metrics['PBIAS_abs']

        best_row = metrics.sort_values(by='SCORE', ascending=False).iloc[0]
        best_model_name = best_row['Model']
        print("✅ Best model:", best_model_name)
        print(best_row[['R2', 'NSE', 'PBIAS']])

        # Plot observed vs predicted
        best_model = best_models[best_model_name]
        y_pred = best_model.predict(X_test)

        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel("Observed")
        plt.ylabel("Simulated")
        plt.title(f"{best_model_name} | R²={best_row['R2']:.2f}, NSE={best_row['NSE']:.2f}, PBIAS={best_row['PBIAS']:.2f}")
        plt.grid()
        plt.tight_layout()
        plt.show()

        return best_model_name, best_models[best_model_name], metrics

    import matplotlib.pyplot as plt

    def predict_discharge_from_climate(
        self,
        new_climate_dataset: xr.Dataset,
        model_name: str,
        model_dir: str = "./Modelos1",
        predictors: list = ["tp", "t2m", "ssrd", "sfcWind"],
        observed_series: pd.Series = None
    ) -> pd.DataFrame:
        """
        Predict monthly discharge using a specified trained model and new climate data.
        Optionally compare predictions with observed data using visual plots.

        Parameters
        ----------
        new_climate_dataset : xr.Dataset
            Monthly climate dataset to use for prediction.
        model_name : str
            Name of the trained model to load (without suffix).
            Example: 'SVR' will load 'SVR_best_model_Flow.joblib'
        model_dir : str, optional
            Directory where trained models and scalers are stored.
        predictors : list of str
            Names of climate variables to use as predictors.
        observed_series : pd.Series, optional
            Observed discharge series for comparison (same index as time in climate data).

        Returns
        -------
        pd.DataFrame
            Predicted discharge time series (monthly).
        """

        # Check predictor variables
        for var in predictors:
            if var not in new_climate_dataset.data_vars:
                raise ValueError(f"Missing predictor variable: {var}")

        # Flatten spatial dimensions
        predictor_arrays = [
            new_climate_dataset[var].data.reshape(-1, new_climate_dataset.sizes["latitude"] * new_climate_dataset.sizes["longitude"])
            for var in predictors
        ]
        X_raw = np.hstack(predictor_arrays).astype(float)

        # Remove NaN columns
        valid_mask = ~np.isnan(X_raw).any(axis=0)
        X_filtered = X_raw[:, valid_mask]

        # Load preprocessing pipeline and scaler
        pipeline_path = os.path.join(model_dir, "pipeline_Flow.pkl")
        scaler_path = os.path.join(model_dir, "scaler_Flow.pkl")
        if not os.path.exists(pipeline_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError("Missing trained pipeline or scaler.")

        pipeline = joblib.load(pipeline_path)
        scaler = joblib.load(scaler_path)

        # Preprocess inputs
        X_scaled = scaler.transform(X_filtered)
        X_eof = pipeline.transform(X_scaled)

        # Load model
        model_file = os.path.join(model_dir, f"{model_name}_best_model_Flow.joblib")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")

        best_model = joblib.load(model_file)

        # Predict
        y_pred = best_model.predict(X_eof)
        time_index = pd.to_datetime(new_climate_dataset.time.data)
        pred_df = pd.DataFrame({"Predicted_Flow": y_pred.flatten()}, index=time_index)

        # Plot comparison if observed_series is provided
        if observed_series is not None:
            common_index = time_index.intersection(observed_series.index)
            observed_series = observed_series.loc[common_index].dropna()
            predicted_series = pred_df["Predicted_Flow"].loc[common_index]

            # Create 3-panel plot
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # 1. Time series
            axes[0].plot(observed_series.index, observed_series.values, label='Observed', color='black')
            axes[0].plot(predicted_series.index, predicted_series.values, label='Predicted', color='blue')
            axes[0].set_title("Time Series Comparison")
            axes[0].set_xlabel("Date")
            axes[0].set_ylabel("Discharge (m³/s)")
            axes[0].legend()
            axes[0].grid()

            # 2. Scatter plot
            axes[1].scatter(observed_series, predicted_series, color='green', alpha=0.6)
            min_val = min(observed_series.min(), predicted_series.min())
            max_val = max(observed_series.max(), predicted_series.max())
            axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
            axes[1].set_title("Observed vs Predicted")
            axes[1].set_xlabel("Observed")
            axes[1].set_ylabel("Predicted")
            axes[1].legend()
            axes[1].grid()

            # 3. Flow Duration Curve (FDC)
            sim_sorted = np.sort(predicted_series.values.flatten())[::-1]
            obs_sorted = np.sort(observed_series.values.flatten())[::-1]

            exceed_sim = np.arange(1, len(sim_sorted) + 1) / len(sim_sorted) * 100
            exceed_obs = np.arange(1, len(obs_sorted) + 1) / len(obs_sorted) * 100

            axes[2].plot(exceed_obs, obs_sorted, label='Observed', color='black')
            axes[2].plot(exceed_sim, sim_sorted, label='Predicted', color='blue')
            axes[2].set_title("Flow Duration Curve")
            axes[2].set_xlabel("Exceedance [%]")
            axes[2].set_ylabel("Discharge (m³/s)")
            axes[2].invert_xaxis()
            axes[2].legend()
            axes[2].grid(True)

            plt.tight_layout()
            plt.show()

        return pred_df








