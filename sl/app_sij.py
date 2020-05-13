import streamlit  as st

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import reverse_geocoder as rg

import math 
import pandas as pd
import numpy as np
import pydeck as pdk


import geopy
import pickle
import datetime
from geopy import distance
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


from keras.models import model_from_json
from keras import callbacks, optimizers



initial_df = pd.read_excel('OUTPUT_WBI_exposer_cyclones_v11.xlsx')


INITIAL_PARAMS = {
    "GENERAL_CATEGORY": ['Cat 4', 'Cat 3', 'Cat 2', 'Cat 5', 'TS', 'Cat 1', 'TD'], 
    "MAX_WIND": 185.0, 
    "MIN_PRES": 1013,
    "BASIN": ['WP', 'NAm', 'SP', 'EP', 'SI', 'NI'],
    "SUB BASIN": 'WP', 
    'MAX_STORMSPEED': 69.0
}

median_cols = ['MAX_STORMSPEED', 
'Arable land (hectares per person)', 
'Food production index (2004-2006 = 100)', 
'GDP per capita (constant 2010 US$)', 
'Life expectancy at birth, total (years)',
'Adjusted savings: education expenditure (% of GNI)',
'Cereal yield (kg per hectare)',
'MIN_PRES',
'RURAL_POP',
'Arable land (hectares per person)']

dummy_cols = ['MONTH_START_1', 'MONTH_START_2', 'MONTH_START_3', 'MONTH_START_4',
       'MONTH_START_5', 'MONTH_START_6', 'MONTH_START_7', 'MONTH_START_8',
       'MONTH_START_9', 'MONTH_START_10', 'MONTH_START_11',
       'MONTH_START_12', 'NEW_GEN_CAT_0', 'NEW_GEN_CAT_2', 'NEW_BASIN_0',
       'NEW_BASIN_2', 'NEW_SUB BASIN_0', 'NEW_SUB BASIN_3',
       'NEW_Income_level_Final_0', 'NEW_Income_level_Final_2', 'SUBGEN_0',
       'SUBGEN_2', 'SUBGEN_3', 'SUBGEN_5', 'SUBINCOME_0', 'SUBINCOME_2',
       'SUBINCOME_3', 'SUBINCOME_5']

object_cols = ['MONTH_START',
 'NEW_GEN_CAT',
 'NEW_BASIN',
 'NEW_SUB BASIN',
 'NEW_Income_level_Final',
 'SUBGEN',
 'SUBINCOME']

features = ['season',
'TOTAL_HOURS_EVENT',
'TOTAL_HOURS_IN_LAND',
'MAX_WIND',
'MIN_PRES',
'MIN_DIST2LAND',
'MAX_STORMSPEED',
'MAX_USA_SSHS',
'MAX_USA_SSHS_INLAND',
'V_LAND_KN',
'POP_DEN_SQ_KM',
'RURAL_POP',
'HDI',
'Arable land (hectares per person)',
'Cereal yield (kg per hectare)',
'Food production index (2004-2006 = 100)',
'GDP per capita (constant 2010 US$)',
'Net flows from UN agencies US$',
'Life expectancy at birth, total (years)',
'Adjusted savings: education expenditure (% of GNI)',
'POP_MAX_34_ADJ',
'POP_MAX_50_ADJ',
'POP_MAX_64_ADJ',
'MAX_SSH_7',
'MAX_SSH_SS',
'Expectancy_break',
'cereal_break',
'cereal_break_two',
'rural_break',
'rural_break_two',
'rural_break_three',
'rural_break_four',
'dis2land_bin',
'pop_break',
'pop_break_two',
'NEW_NATURE',
'MONTH_START_1',
'MONTH_START_2',
'MONTH_START_3',
'MONTH_START_4',
'MONTH_START_5',
'MONTH_START_6',
'MONTH_START_7',
'MONTH_START_8',
'MONTH_START_9',
'MONTH_START_10',
'MONTH_START_11',
'MONTH_START_12',
'NEW_GEN_CAT_0',
'NEW_GEN_CAT_2',
'NEW_BASIN_0',
'NEW_BASIN_2',
'NEW_SUB BASIN_0',
'NEW_SUB BASIN_3',
'NEW_Income_level_Final_0',
'NEW_Income_level_Final_2',
'SUBGEN_0',
'SUBGEN_2',
'SUBGEN_3',
'SUBGEN_5',
'SUBINCOME_0',
'SUBINCOME_2',
'SUBINCOME_3',
'SUBINCOME_5']

log_cols = ['TOTAL_HOURS_EVENT', 'TOTAL_HOURS_IN_LAND', 
            'POP_DEN_SQ_KM', 'Arable land (hectares per person)']

def new_cats(df):
### Final Trans


    df['NEW_GEN_CAT'] = np.where(df['GENERAL_CATEGORY'] == 'Cat 1', 1, 0)
    df['NEW_GEN_CAT'] = np.where(df['GENERAL_CATEGORY'] == 'TS', 2, 0)
    df['NEW_GEN_CAT'] = df['NEW_GEN_CAT'].astype('object')


    df['NEW_NATURE'] = np.where(df['NATURE'] == 'TS', 1, 0)
    df['NEW_NATURE'] = np.where(df['NATURE'] == 'ET', 1, 0)

    df['NEW_BASIN'] = np.where(df['BASIN'] == 'SP', 1, 0)
    df['NEW_BASIN'] = np.where(df['BASIN'] == 'WP', 2, 0)
    df['NEW_BASIN'] = df['NEW_BASIN'].astype('object')


    df['NEW_SUB BASIN'] = np.where(df['SUB_BASIN'] == 'EP', 1, 0)
    df['NEW_SUB BASIN'] = np.where(df['SUB_BASIN'] == 'NAm', 2, 0)
    df['NEW_SUB BASIN'] = np.where(df['SUB_BASIN'] == 'CS', 3, 0)
    df['NEW_SUB BASIN'] = df['NEW_SUB BASIN'].astype('object')


    df['NEW_Income_level_Final'] = np.where(df['Income_level_Final'] == 'Low_Middle', 1, 0)
    df['NEW_Income_level_Final'] = np.where(df['Income_level_Final'] == 'High_Middle', 2, 0)
    df['NEW_Income_level_Final'] = df['NEW_Income_level_Final'].astype('object')


    df["SUBGEN"] = df['NEW_SUB BASIN'] + df['NEW_GEN_CAT']
    df["SUBINCOME"] = df['NEW_SUB BASIN'] + df['NEW_Income_level_Final']


def new_cont(df):

    df['Expectancy_break'] = np.where(
        df['Life expectancy at birth, total (years)'] > 67, 1, 0)


    df["cereal_break"] = np.where(df['Cereal yield (kg per hectare)'].between(2500, 5000), 1, 0)
    df["cereal_break_two"] = np.where(df['Cereal yield (kg per hectare)'] > 3650.0, 1, 0)


    df["rural_break"] = np.where(df['RURAL_POP'] < 37.5, 1, 0)
    df["rural_break_two"] = np.where(df['RURAL_POP'].between(37.5, 62.5), 1, 0)
    df["rural_break_three"] = np.where(df['RURAL_POP'] > 62.5, 1, 0)


    df["rural_break_four"] = np.where(df['RURAL_POP'] > 37.5, 1, 0)


    df['dis2land_bin'] = np.where(df['MIN_DIST2LAND'] == 0, 1, 0)

    df["pop_break"] = np.where(df['POP_DEN_SQ_KM'].between(1, 2), 1, 0)
    df["pop_break_two"] = np.where(df['POP_DEN_SQ_KM'].between(2, 3), 1, 0)



def get_iso(coords):
    return rg.search(coords, mode = 1)[0]["cc"]

## Rerun button - 
st.button("Re Run")

#sidebar text asking for upload
st.sidebar.title("Upload")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# what day is today?
dt = datetime.datetime.today()
#what month?
m =  dt.month

st.sidebar.title("Upload tracks to calculate exposer exposer")
st.sidebar.markdown("Upload file with coordinates and radii of cyclone, coordinates and radii of 34, 50, 64 knots per hour, NAMES: LAT, LON, R34, R50, R50")


#default upload for now
#DATA_URL = ("is.csv")

@st.cache(persist=True, allow_output_mutation=True, suppress_st_warning=True)

#loading clients track
def load_data(data):

    
    data = pd.read_csv(data)
    data['ISO_TIME'] = pd.to_datetime(data['ISO_TIME'])

    data['YEAR'] = data['ISO_TIME'].dt.year
   
    #data = data.dropna()
    data['SID']=1

    lowercase = lambda x: str(x).lower()
   # data.dropna(subset=['LAT', 'LON', 'USA_R34_NE'], inplace=True)
    
    data.rename(columns={'LAT': 'lat'},inplace=True)
    data.rename(columns={'LON': 'lon'},inplace=True)
    data.rename(lowercase, axis='columns', inplace=True)


    return data



if uploaded_file is not None:
    data = load_data(uploaded_file)

    #data = data.dropna()

    

    data["COORDS"] = data[["lat", "lon"]].values.tolist()
    data["COORDS"] = data['COORDS'].apply(lambda x: (x[0], x[1]))
    data["ISO"] = data['COORDS'].apply(get_iso)

    paths = data.copy()



    
    MONTH_START = st.sidebar.text_input("MONTH_START", m)
    GENERAL_CATEGORY = st.sidebar.selectbox("Select GENERAL_CATEGORY", INITIAL_PARAMS["GENERAL_CATEGORY"])
    MAX_WIND = st.sidebar.text_input("MAX_WIND", INITIAL_PARAMS["MAX_WIND"])
    MIN_PRES = st.sidebar.text_input("MIN_PRES", INITIAL_PARAMS["MIN_PRES"])
    BASIN = st.sidebar.selectbox("BASIN", INITIAL_PARAMS["BASIN"])

    SUB_BASIN = st.sidebar.text_input("SUB BASIN", INITIAL_PARAMS["SUB BASIN"])
    MIN_DIST2LAND    = st.sidebar.text_input("MIN_DIST2LAND")
    MAX_STORMSPEED = st.sidebar.text_input("MAX_STORMSPEED", INITIAL_PARAMS["MAX_STORMSPEED"])
    TOTAL_HOURS_EVENT = st.sidebar.text_input("TOTAL_HOURS_EVENT")
    TOTAL_HOURS_IN_LAND = st.sidebar.text_input("TOTAL_HOURS_IN_LAND")
    NATURE = st.sidebar.text_input("NATURE", 5)
    


    paths['MONTH_START'] = MONTH_START
    paths['MIN_PRES'] = MIN_PRES
    paths['BASIN'] = BASIN

    paths['SUB_BASIN'] = SUB_BASIN
    paths['MAX_STORMSPEED'] = MAX_STORMSPEED
    paths['NATURE'] = NATURE
    paths['MAX_WIND'] = MAX_WIND
    paths['MIN_PRES'] = MIN_PRES

    paths['TOTAL_HOURS_EVENT'] = TOTAL_HOURS_EVENT
    paths['TOTAL_HOURS_IN_LAND'] = TOTAL_HOURS_IN_LAND

    paths['MIN_DIST2LAND'] = MIN_DIST2LAND


    paths['GENERAL_CATEGORY'] = GENERAL_CATEGORY

    ## Temp values for pop_max
    paths['POP_MAX_34_ADJ'] = 1
    paths['POP_MAX_50_ADJ'] = 1
    paths['POP_MAX_64_ADJ'] = 1
    


    if st.checkbox("Show Raw data", False):
        st.subheader('Raw Data')
        st.write(paths)




    ### Pre Processing

    wbi = pd.read_csv("result-wbi.csv")
    df = pd.merge(paths, wbi, left_on=["ISO", "season"], right_on=['Alpha-2 code', 'Year'], how='left')
    


    df["SUB_BASIN"]= df["SUB_BASIN"].replace('MM', np.nan) 
    df["BASIN"]= df["BASIN"].replace('MM', np.nan)
    df['SUB_BASIN']= np.where(df['SUB_BASIN'].isnull(), df['BASIN'], df['SUB_BASIN'])


    GB_model = pickle.load(open('GBM.model', 'rb'))
    XGB_model = pickle.load(open('XGB.model', 'rb'))

    median_imputer = pickle.load(open('median.imp', 'rb'))
    OHenc = pickle.load(open('enc.encoder', 'rb'))
    scale = pickle.load(open('scale.scaler', 'rb'))

    kmeans = pickle.load(open('kmeans.cluster', 'rb'))

    ## Load Keras model
    json_file = open('keras.json', 'r')
    keras_model_json = json_file.read()
    json_file.close()

    keras_model = model_from_json(keras_model_json)

    optim = optimizers.RMSprop(learning_rate=0.001)
    keras_model.compile(loss='mse', optimizer=optim, metrics=['mae'])

    ## transform median imp
    df[median_cols] = median_imputer.transform(df[median_cols])

    new_cats(df)
    new_cont(df)


    temp = OHenc.transform(df[object_cols])
    dummies = pd.DataFrame(data=temp, columns=dummy_cols)
    df = pd.concat([df, dummies], axis=1)

    df = df.fillna(100)

    X = df[features]

    X = scale.transform(X)


    km_pred = kmeans.predict(X).reshape(-1, 1)
    X = np.column_stack((X, km_pred))
    

    ## Kmeans add labels column



    st.write('pred_data', X)


select_model = st.selectbox("Select model",["GBM","XGBoost","NN"])

if select_model == "GBM":
    pred = GB_model.predict(X)
    st.write('Predict', pred**2)

if select_model == "NN":
    pred = keras_model.predict(X)
    st.write('Predict', pred**2)
    
if select_model == "XGBoost":
    pred = XGB_model.predict(X)
    st.write('Predict', pred**2)


'''
select_model = st.selectbox("Select model",["ensembled","svr","nn"])

def esmld():
    pass

    st.title(er.fit(X_train, y_train).predict(X_test))

if st.button("Generate Score for metrics model"):
    if  select_model == 'ensembled':
        esmld()
        st.balloons()
    if uploaded_file is not None:
        if st.button("Generate prediction for uploaded data"):
            if  select_model == 'ensembled':
               passs
               st.balloons()
'''
