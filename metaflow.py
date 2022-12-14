import pandas as pd
import numpy as np
import geopandas as gpd
import osmnx as ox
import contextily as ctx
from geopandas import GeoDataFrame
from shapely.geometry import Point, MultiPoint
from shapely.ops import nearest_points
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm, tqdm_notebook
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform
from sklearn.preprocessing import StandardScaler 
from pprint import pprint
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from metaflow import step, FlowSpec, Parameter, JSONType

class MyFlow(FlowSpec):
    models = Parameter(
        "models",
        help=("A list of model class to train and test."),
        type=JSONType,
        default='["Random Forest", "Gradient Boosting"]',
    )
    @step
    def start(self):
        self.next(self.load_data)

    @step
    def load_data(self):
        self.df_1 = pd.read_csv("Home_n_Map.csv")
        gpd_1_degree = gpd.GeoDataFrame(self.df_1, geometry=gpd.points_from_xy(self.df_1.Longitude, self.df_1.Latitude), crs={'init' :'epsg:4326'})
        fig, ax = plt.subplots(figsize=(12, 10))
        gpd_1_degree.to_crs(epsg=3857).plot(ax = ax,
                                        figsize=(12,12),
                                        markersize=40,
                                        color="black",
                                        edgecolor="white",
                                        alpha=0.8,
                                        marker="o"
                                        )
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
        self.next(self.clean_data)

    @step
    def clean_data(self):
        self.df_1 = self.df_1.rename(columns={'NO value': 'NO'})
        self.df_1 = self.df_1.rename(columns={'NO2 value': 'NO2'})
        self.df_1 = self.df_1.rename(columns={'PM2p5 value': 'PM2p5'})
        print("*** Cnts of Each Feature ***")
        print(self.df_1.nunique())
        self.df_1_drop = self.df_1.drop(['state', 'county', 'tract_name', 'GEOID'], axis=1) # Drop irrelevant feature
        # Missing values
        print("*** Missing Values ***")
        print('Sum of N/A values', self.df_1_drop.isnull().sum())
        self.df_1_miss = self.df_1_drop.dropna(axis=0, subset=['zone', 'wind', 'temp']) # Here we choose to drop all N/A
        self.df_1_miss = self.df_1_miss.reset_index()
        print(self.df_1_miss.info()) # Check information
        self.df_1_miss = self.df_1_miss.drop(['index'], axis=1) # Remove index
        # Duplicated values
        self.df_1_dup = self.df_1_miss.drop_duplicates() # Drop
        self.df_1_dup.reset_index(inplace=True)
        # Outliers
        _,axss = plt.subplots(2,3, figsize=[20,10])  # create a 2x3 matrix = 6 figures
        self.df_1_out = self.df_1_dup.copy()
        sns.boxplot(y ='NO', data = self.df_1_out, ax=axss[0, 0])
        sns.boxplot(y ='NO2', data = self.df_1_out, ax=axss[0, 1])
        sns.boxplot(y ='PM2p5', data = self.df_1_out, ax=axss[0, 2])
        sns.boxplot(y ='pop_den', data = self.df_1_out, ax=axss[1][0])
        sns.boxplot(y ='wind', data = self.df_1_out, ax=axss[1][1])
        sns.boxplot(y ='temp', data = self.df_1_out, ax=axss[1][2])
        self.df_1_out.loc[self.df_1_out['NO'] > 200, 'NO'] = 200  # If NO > 200 -> let all of them equl to 200
        self.next(self.feature_engineering)

    @step
    def feature_engineering(self):
        Oakland_poly = ox.geocode_to_gdf('Oakland, California')
        #print(Oakland_poly.plot())
        gpd_1_degree = gpd.GeoDataFrame(self.df_1_out, geometry = self.df_1_out['geometry'], crs={'init' :'epsg:4326'})
        Oakland_poly.crs, gpd_1_degree.crs
        gpd_1_city = gpd.sjoin(gpd_1_degree, Oakland_poly, how="inner", op="intersects")
        gpd_1_city = gpd_1_city.drop(['index_right', 'bbox_east', 'bbox_north', 'bbox_south', 'bbox_west'], axis=1)

        #city
        # grab street data (roads and intersections) for entire city
        oak_streets = ox.graph_from_place('Oakland, California', network_type = 'drive')
        nodes, edges = ox.graph_to_gdfs(oak_streets)
        #edges.plot()
        oakland_rds = edges.copy()
        oakland_rds['highway'] = oakland_rds['highway'].str.replace('_link', '')
        oakland_rds['highway'] = np.where(oakland_rds['highway'] == 'trunk', 'secondary', oakland_rds['highway'])
        oakland_rds['highway'] = np.where(oakland_rds['highway'] == 'living_street', 'residential', oakland_rds['highway'])
        #sns.countplot(oakland_rds['highway'])
    
        oakland_highways = oakland_rds[oakland_rds.highway == 'motorway']
        oakland_primary = oakland_rds[oakland_rds.highway == 'primary']
        oakland_secondary = oakland_rds[oakland_rds.highway == 'secondary']
        oakland_tertiary = oakland_rds[oakland_rds.highway == 'tertiary']
        oakland_resid = oakland_rds[oakland_rds.highway == 'residential']
        #oakland_highways.crs

        # fig, ax = plt.subplots(figsize=(12, 10))
        # oakland_highways.to_crs(epsg=3857).plot(ax = ax, figsize=(12,12), markersize=40, color="red", edgecolor="white", alpha=0.8, marker="o")
        # ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

        '''
        fig, ax = plt.subplots(figsize=(12, 10))
        oakland_primary.to_crs(epsg=3857).plot(ax = ax,
                                            figsize=(12,12),
                                            markersize=40,
                                            color="blue",
                                            edgecolor="white",
                                            alpha=0.8,
                                            marker="o"
                                            )
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

        fig, ax = plt.subplots(figsize=(12, 10))
        oakland_secondary.to_crs(epsg=3857).plot(ax = ax,
                                            figsize=(12,12),
                                            markersize=40,
                                            color="green",
                                            edgecolor="white",
                                            alpha=0.8,
                                            marker="o"
                                            )
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

        fig, ax = plt.subplots(figsize=(12, 10))
        oakland_primary.to_crs(epsg=3857).plot(ax = ax,
                                            figsize=(12,12),
                                            markersize=40,
                                            color="blue",
                                            edgecolor="white",
                                            alpha=0.8,
                                            marker="o"
                                            )
        oakland_secondary.to_crs(epsg=3857).plot(ax = ax,
                                            figsize=(12,12),
                                            markersize=40,
                                            color="green",
                                            edgecolor="white",
                                            alpha=0.8,
                                            marker="o"
                                            )

        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

        fig, ax = plt.subplots(figsize=(12, 10))
        oakland_tertiary.to_crs(epsg=3857).plot(ax = ax,
                                            figsize=(12,12),
                                            markersize=40,
                                            color="purple",
                                            edgecolor="white",
                                            alpha=0.8,
                                            marker="o"
                                            )
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

        print(gpd_1_city.crs)
        '''
        gpd_1_city_utm = gpd_1_city.to_crs({'init': 'epsg:32610'}).copy()  
        highway_utm = oakland_highways.to_crs({'init': 'epsg:32610'}).copy()
        primary_utm = oakland_primary.to_crs({'init': 'epsg:32610'}).copy()
        secondary_utm = oakland_secondary.to_crs({'init': 'epsg:32610'}).copy()
        tertiary_utm = oakland_tertiary.to_crs({'init': 'epsg:32610'}).copy()

        # UDF
        def distance_to_roadway(gps, roadway):
            dists = []
            for i in roadway.geometry:
                dists.append(i.distance(gps))
            return(np.min(dists))

        # Calculate distance to nearest highway
        tqdm.pandas()
        gpd_1_city['closest_highway'] = gpd_1_city_utm['geometry'].progress_apply(distance_to_roadway, roadway = highway_utm)

        # Calculate distance to nearest primary road
        tqdm.pandas()
        gpd_1_city['closest_primary'] = gpd_1_city_utm['geometry'].progress_apply(distance_to_roadway, roadway = primary_utm)

        # Calculate distance to nearest secondary road
        tqdm.pandas()
        gpd_1_city['closest_secondary'] = gpd_1_city_utm['geometry'].progress_apply(distance_to_roadway, roadway = secondary_utm)

        # Calculate distance to nearest tertiary road
        tqdm.pandas()
        gpd_1_city['closest_tertiary'] = gpd_1_city_utm['geometry'].progress_apply(distance_to_roadway, roadway = tertiary_utm)

        #City Structure: traffic signal & stop sign
        trafic_signals = nodes[nodes['highway'] == 'traffic_signals']
        stop_cross = nodes[nodes['highway'] == 'stop']
        '''
        fig, ax = plt.subplots(figsize=(12, 10))
        trafic_signals.to_crs(epsg=3857).plot(ax = ax,
                                            figsize=(12,12),
                                            markersize=40,
                                            color="blue",
                                            edgecolor="white",
                                            alpha=0.8,
                                            marker="o"
                                            )
        stop_cross.to_crs(epsg=3857).plot(ax = ax,
                                            figsize=(12,12),
                                            markersize=40,
                                            color="orangered",
                                            edgecolor="white",
                                            alpha=0.8,
                                            marker="o"
                                            )
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
        '''
        traffic_sig_utm = trafic_signals.to_crs({'init': 'epsg:32610'}).copy() 
        stop_sign_utm = stop_cross.to_crs({'init': 'epsg:32610'}).copy()      

        def nearest_intersection(gps, intersections):
            ''' Calculates distance from GPS point to nearest intersection'''
            closest_point = nearest_points(gps, MultiPoint(intersections.values))[1]
            return(gps.distance(closest_point))
        
        tqdm.pandas()
        gpd_1_city['trafic_signal_dist'] = gpd_1_city_utm['geometry'].progress_apply(nearest_intersection, intersections = traffic_sig_utm['geometry'])

        tqdm.pandas()
        gpd_1_city['stop_sign_dist'] = gpd_1_city_utm['geometry'].progress_apply(nearest_intersection, intersections = stop_sign_utm['geometry'])

        # Category Encoding
        zone = gpd_1_city['zone']
        road_type = gpd_1_city['road_type']

        # One-hot encoding
        gpd_1_city = pd.get_dummies(gpd_1_city, columns=['road_type'], drop_first=False) 
        gpd_1_city = pd.get_dummies(gpd_1_city, columns=['zone'], drop_first=False)
        gpd_1_city = pd.concat([gpd_1_city, zone], axis = 1)
        gpd_1_city = pd.concat([gpd_1_city, road_type], axis = 1)

        # Data Preparation
        gpd_1_city = gpd_1_city.drop(['index', 'Pt_CANCR'], axis=1)
        gpd_1_city['Respiratory_HI'] = np.where(gpd_1_city['Respiratory_HI'].str.contains('high'), '3', gpd_1_city['Respiratory_HI'])
        gpd_1_city['Respiratory_HI'] = np.where(gpd_1_city['Respiratory_HI'].str.contains('moderate'), '2', gpd_1_city['Respiratory_HI'])
        gpd_1_city['Respiratory_HI'] = np.where(gpd_1_city['Respiratory_HI'].str.contains('low'), '1', gpd_1_city['Respiratory_HI'])
        gpd_1_city["Respiratory_HI"] = gpd_1_city.Respiratory_HI.astype(float)

        # Numerical Features
        self.numerical = ['NO', 'NO2', 'PM2p5', 'pop_den', 'wind', 'temp', 'closest_highway', 'closest_primary', 'closest_secondary', 'closest_tertiary', 
                    'trafic_signal_dist', 'stop_sign_dist', 'road_type_motorway', 'road_type_primary', 'road_type_residential', 'road_type_secondary', 'road_type_tertiary', 
                    'road_type_unclassified', 'zone_commercial', 'zone_industrial', 'zone_mixed', 'zone_open_space', 'zone_residential', 'Longitude', 'Latitude', 'Respiratory_HI']

        # Categorical Features 
        self.categorical = ['geometry', 'zone', 'road_type']
        self.gpd_1_city = gpd_1_city
        self.next(self.data_visualization)

    @step
    def data_visualization(self):

        df_vis = self.gpd_1_city.copy()

        f, ax = plt.subplots(figsize= [20,15])
        sns.heatmap(df_vis[self.numerical].corr(), annot=True, fmt=".2f", ax=ax, cmap = "magma" )
        ax.set_title("Correlation Matrix", fontsize=20)
        plt.show()

        Corr = pd.DataFrame(df_vis[self.numerical].corr()['Respiratory_HI'].sort_values(ascending=False))
        Corr = Corr.iloc[1:,:]
        Corr.columns=['Target Correlation']
        g0 = sns.barplot(x="Target Correlation", y=Corr.index, data=Corr)
        g0.figure.set_size_inches(12, 9)

        gpd_1_vis = self.gpd_1_city.copy()

        plt.figure(figsize = (11, 10))
        plt.scatter(gpd_1_vis.Longitude, gpd_1_vis.Latitude, s=5, c = gpd_1_vis.NO2)
        plt.colorbar()
        plt.xlabel('Longitude', fontsize=18)
        plt.ylabel('Latitude', fontsize=18)

        plt.figure(figsize = (11, 10))
        plt.scatter(gpd_1_vis.Longitude, gpd_1_vis.Latitude, s=5, c = gpd_1_vis.PM2p5, cmap='inferno')
        plt.colorbar()
        plt.xlabel('Longitude', fontsize=18)
        plt.ylabel('Latitude', fontsize=18)
        self.df_vis = df_vis
        self.next(self.prepare_train_and_test_dataset)

    @step
    def prepare_train_and_test_dataset(self):
        self.df_model = self.df_vis.copy()
        self.X = self.df_model[self.numerical].drop(['Respiratory_HI', 'Longitude','Latitude', 'road_type_unclassified','NO', 'NO2', 'PM2p5'], axis=1)

        # NO
        self.y_NO = self.df_model['NO']
        self.X_train_NO, self.X_test_NO, self.y_train_NO, self.y_test_NO = model_selection.train_test_split(self.X, self.y_NO, test_size=0.25, random_state= 1)
        scaler1 = StandardScaler()                               
        scaler1.fit(self.X_train_NO) 
        self.X_train_NO = pd.DataFrame(scaler1.transform(self.X_train_NO))                             
        self.X_test_NO = pd.DataFrame(scaler1.transform(self.X_test_NO))
        self.no_train = [self.X_train_NO, self.y_train_NO]
        self.no_test = [self.X_test_NO, self.y_test_NO]
        self.no = [self.no_train, self.no_test]

        # NO2
        self.y_NO2 = self.df_model['NO2']
        self.X_train_NO2, self.X_test_NO2, self.y_train_NO2, self.y_test_NO2 = model_selection.train_test_split(self.X, self.y_NO2, test_size=0.25, random_state= 1)
        scaler2 = StandardScaler()                               
        scaler2.fit(self.X_train_NO2)                                   
        self.X_train_NO2 = pd.DataFrame(scaler2.transform(self.X_train_NO2))              
        self.X_test_NO2 = pd.DataFrame(scaler2.transform(self.X_test_NO2))
        self.no2_train = [self.X_train_NO2, self.y_train_NO2]
        self.no2_test = [self.X_test_NO2, self.y_test_NO2]
        self.no2 = [self.no2_train, self.no2_test]

        # PM   
        self.y_PM = self.df_model['PM2p5']
        self.X_train_PM, self.X_test_PM, self.y_train_PM, self.y_test_PM = model_selection.train_test_split(self.X, self.y_PM, test_size=0.25, random_state= 1)
        scaler3 = StandardScaler()                         
        scaler3.fit(self.X_train_PM)                            
        self.X_train_PM = pd.DataFrame(scaler3.transform(self.X_train_PM))         
        self.X_test_PM = pd.DataFrame(scaler3.transform(self.X_test_PM))
        self.pm_train = [self.X_train_PM, self.y_train_PM]
        self.pm_test = [self.X_test_PM, self.y_test_PM]
        self.pm = [self.pm_train, self.pm_test]

        
        
        self.next(self.train_model, foreach = "models")

    @step
    def train_model(self):

        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.ensemble import RandomForestRegressor

        model_name = self.input

        if model_name == "Random Forest":
            self.model = RandomForestRegressor(n_jobs=2)
            self.params = {'max_features': [6, 8, 10],
                    'n_estimators': [150, 200]}
        elif model_name == "Gradient Boosting":
            self.model = GradientBoostingRegressor()
            self.params = {'max_features': [6, 8, 10],
                    'learning_rate': [0.05, 0.1, 0.5],
                    'n_estimators': [150, 200]}
        else:
            raise ValueError("Invalid data name")
            
        def pred_summary(pred, ytest, limit = 200):    
            print('RMSE', np.sqrt(mean_squared_error(ytest, pred)))
            print('R2', r2_score(ytest, pred))


        self.model_grid_no = GridSearchCV(self.model, self.params, cv=5, scoring = 'neg_mean_squared_error')
        self.model_grid_no.fit(self.no[0][0], self.no[0][1])
        print('Best score (RMSE)', np.sqrt(np.abs(self.model_grid_no.best_score_)))
        print(self.model_grid_no.best_estimator_)
        self.model_out_no = self.model_grid_no.predict(self.no[1][0])
        pred_summary(self.model_out_no, self.no[1][1], limit=50)
        self.FI_no = pd.DataFrame(self.model_grid_no.best_estimator_.feature_importances_, index = self.no[0][0].columns, columns=['Feature Importance'])
        self.FI_no = self.FI_no.sort_values(by = 'Feature Importance',ascending=False)
        print(self.FI_no)

        self.summary_no = pd.DataFrame({'Random Forest':list(self.FI_no.index),
                    'Gradient Boost':list(self.FI_no.index)})
        print(self.summary_no)


        self.model_grid_no2 = GridSearchCV(self.model, self.params, cv=5, scoring = 'neg_mean_squared_error')
        self.model_grid_no2.fit(self.no2[0][0], self.no2[0][1])
        print('Best score (RMSE)', np.sqrt(np.abs(self.model_grid_no2.best_score_)))
        print(self.model_grid_no2.best_estimator_)
        self.model_out_no2 = self.model_grid_no2.predict(self.no2[1][0])
        pred_summary(self.model_out_no2, self.no2[1][1], limit=50)
        self.FI_no2 = pd.DataFrame(self.model_grid_no2.best_estimator_.feature_importances_, index = self.no2[0][0].columns, columns=['Feature Importance'])
        self.FI_no2 = self.FI_no2.sort_values(by = 'Feature Importance',ascending=False)
        print(self.FI_no2)

        self.summary_no2 = pd.DataFrame({'Random Forest':list(self.FI_no2.index),
                    'Gradient Boost':list(self.FI_no2.index)})
        print(self.summary_no2)


        self.model_grid_pm = GridSearchCV(self.model, self.params, cv=5, scoring = 'neg_mean_squared_error')
        self.model_grid_pm.fit(self.pm[0][0], self.pm[0][1])
        print('Best score (RMSE)', np.sqrt(np.abs(self.model_grid_pm.best_score_)))
        print(self.model_grid_pm.best_estimator_)
        self.model_out_pm = self.model_grid_pm.predict(self.pm[1][0])
        pred_summary(self.model_out_pm, self.pm[1][1], limit=50)
        self.FI_pm = pd.DataFrame(self.model_grid_pm.best_estimator_.feature_importances_, index = self.pm[0][0].columns, columns=['Feature Importance'])
        self.FI_pm = self.FI_pm.sort_values(by = 'Feature Importance',ascending=False)
        print(self.FI_pm)

        self.summary_pm = pd.DataFrame({'Random Forest':list(self.FI_pm.index),
                    'Gradient Boost':list(self.FI_pm.index)})
        print(self.summary_pm)



        self.RMSE_no = np.sqrt(mean_squared_error(self.model_out_no, self.no[1][1]))
        self.RMSE_no2 = np.sqrt(mean_squared_error(self.model_out_no2, self.no2[1][1]))
        self.RMSE_pm = np.sqrt(mean_squared_error(self.model_out_pm, self.pm[1][1]))

        self.next(self.join)

    @step
    def join(self, inputs):

        self.best_model_no, self.best_score_no = None, float("-inf")
        self.best_model_no2, self.best_score_no2 = None, float("-inf")
        self.best_model_pm, self.best_score_pm = None, float("-inf")
        for _input in inputs:
            if _input.RMSE_no > self.best_score_no:
                self.best_score_no = _input.RMSE_no
                self.best_model_no = _input.model_grid_no
            if _input.RMSE_no2 > self.best_score_no2:
                self.best_score_no2 = _input.RMSE_no2
                self.best_model_no2 = _input.model_grid_no2
            if _input.RMSE_pm > self.best_score_pm:
                self.best_score_pm = _input.RMSE_pm
                self.best_model_pm = _input.model_grid_pm

        self.next(self.end)


    @step
    def end(self):
        pass

if __name__ == "__main__":
    MyFlow()
