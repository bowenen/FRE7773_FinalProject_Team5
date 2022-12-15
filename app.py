import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from metaflow import Flow
from metaflow import get_metadata, metadata
import matplotlib.pyplot as plt
import pandas as pd
FLOW_NAME = 'MyFlow'
metadata('/home/bowenen/')
print(get_metadata())
 
@st.cache
def get_latest_successful_run(flow_name: str):
    "Gets the latest successfull run."
    for r in Flow(flow_name).runs():
        if r.successful: 
            return r

latest_run = get_latest_successful_run(FLOW_NAME)
latest_model_no = latest_run.data.best_model_no
latest_model_no2 = latest_run.data.best_model_no2
latest_model_pm = latest_run.data.best_model_pm


# show predictions
st.markdown("## Predictions")

# play with the model
st.markdown("## Model")
_x1 = st.text_input('closest_highway:', 3)
_x2 = st.text_input('wind: ', 2)
_x3 = st.text_input('road_type_motorway: ', 1)
_x = {
    'pop_den':[7],
    'wind': [_x2],
    'temp':[14],
    'closest_highway': [_x1],
    'closest_primary':[25],
    'closest_secondary':[5],
    'closest_tertiary':[5],
    'trafic_signal_dist':[10],
    'stop_sign_dist':[15],
    'road_type_motorway': [_x3],
    'road_type_primary':[0],
    'zone_residential':[0],
    'road_type_secondary':[0],
    'road_type_tertiary':[0],
    'zone_commercial':[0],
    'zone_industrial':[0],
    'zone_mixed':[0],
    'zone_open_space':[0],
    'road_type_residential':[0]
}
_x = pd.DataFrame(_x)
val_no = latest_model_no.predict(_x)
val_no2 = latest_model_no2.predict(_x)
val_pm = latest_model_pm.predict(_x)
st.write('Inputs are: closest_highway: {}, wind: {}, road_type_motorway: {}'.format(_x1, _x2, _x3))
st.write('NO prediction is {}, NO2 prediction is {}, PM2.5 prediction is {}'.format(val_no, val_no2, val_pm))
st.markdown("## Feature Importance")
FI_no = pd.DataFrame(latest_model_no.best_estimator_.feature_importances_, index = _x.columns, columns=['Feature Importance'])
FI_no = FI_no.sort_values(by = 'Feature Importance',ascending=False)
FI_no2 = pd.DataFrame(latest_model_no2.best_estimator_.feature_importances_, index = _x.columns, columns=['Feature Importance'])
FI_no2 = FI_no2.sort_values(by = 'Feature Importance',ascending=False)
FI_pm = pd.DataFrame(latest_model_pm.best_estimator_.feature_importances_, index = _x.columns, columns=['Feature Importance'])
FI_pm = FI_pm.sort_values(by = 'Feature Importance',ascending=False)
summary_ = pd.DataFrame({'NO':list(FI_no.index),
            'NO2':list(FI_no2.index),
            'PM2.5':list(FI_pm.index)})
st.write(summary_)

# python3 -m streamlit run app.py