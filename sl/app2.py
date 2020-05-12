## Importing the required libraries
import pandas as pd
import numpy as np
import pickle
import datetime
import os
import reverse_geocoder as rg
import math
from geopy import distance
#### Sklearn Libraries
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.externals import joblib
### stream Lit UI
import streamlit as st
import pydeck as pdk

# #### Global Variables #####
# one_encoder_GENERAL_CATEGORY = OneHotEncoder()
# one_encoder_NATURE = OneHotEncoder()
# one_encoder_MONTH_START = OneHotEncoder()
# one_encoder_SUB_BASIN = OneHotEncoder()
########### 1 . Model selection and training ###########
def preprocessing(input_file=None):
    # global one_encoder_GENERAL_CATEGORY, one_encoder_MONTH_START, one_encoder_NATURE, one_encoder_SUB_BASIN
    one_encoder_GENERAL_CATEGORY = OneHotEncoder()
    one_encoder_NATURE = OneHotEncoder()
    one_encoder_MONTH_START = OneHotEncoder()
    one_encoder_SUB_BASIN = OneHotEncoder()
    if input_file == None:
        input_file = 'OUTPUT_WBI_exposer_cyclones_v11.xlsx'
    init_data = pd.read_excel(input_file)
    init_data = init_data.drop_duplicates(subset=['NAME','ISO','YEAR'], keep="last")

    init_data['COORDS_both']=init_data['COORDS'].str[2:18]
    init_data['COORDS_1'] = init_data['COORDS_both'].str.split(", ", n = 1, expand = True)[0]
    init_data['COORDS_2'] = init_data['COORDS_both'].str.split(", ",  expand = True)[1]
    init_data['COORDS_2'] = init_data['COORDS_2'].str.split(")",  expand = True)[0]
    init_data['COORDS_1'] = init_data['COORDS_1'].astype(float)
    init_data['COORDS_2'] = init_data['COORDS_2'].astype(float)

    init_data['POP_MAX_50_ADJ'] = init_data['POP_MAX_50_ADJ']-init_data['POP_MAX_34_ADJ']
    init_data['POP_MAX_50'] = init_data['POP_MAX_50']-init_data['POP_MAX_34']
    init_data['POP_MAX_34_ADJ'] = init_data['POP_MAX_34_ADJ']-init_data['POP_MAX_50_ADJ']
    init_data['POP_MAX_34'] = init_data['POP_MAX_34']-init_data['POP_MAX_50']
    init_data['POP_MAX_50'][init_data['POP_MAX_50'] < 0] = 0
    init_data['POP_MAX_50_ADJ'][init_data['POP_MAX_50_ADJ'] < 0] = 0
    init_data['POP_MAX_64'][init_data['POP_MAX_64'] < 0] = 0
    init_data['POP_MAX_64_ADJ'][init_data['POP_MAX_64_ADJ'] < 0] = 0


    req_columns = [ 'SUB BASIN', 'ISO', 'YEAR', 'MONTH_START',
       'TOTAL_HOURS_IN_LAND', 'NATURE',
       'GENERAL_CATEGORY', 'MAX_WIND', 'MIN_PRES',
       'MAX_STORMSPEED','DISTANCE_TRACK_VINCENTY', 
       'POP_MAX_34_ADJ', 'POP_MAX_50_ADJ', 'POP_MAX_64_ADJ', 
       'POP_DEN_SQ_KM', 'RURAL_POP(%)', 'POP_TOTAL', 'RURAL_POP', 
       'HDI',
       'TOTAL_AFFECTED', 
         'COORDS_1', 'COORDS_2']
    df = init_data[req_columns]
    # pd.DataFrame(ohe.transform(x_test).toarray(), columns = ohe.get_feature_names())

    df_GC = pd.DataFrame(one_encoder_GENERAL_CATEGORY.fit_transform(df[['GENERAL_CATEGORY']]).toarray(),
    columns=one_encoder_GENERAL_CATEGORY.get_feature_names())
    df_NA = pd.DataFrame(one_encoder_NATURE.fit_transform(df[['NATURE']]).toarray(),
    columns=one_encoder_NATURE.get_feature_names())
    df_MS = pd.DataFrame(one_encoder_MONTH_START.fit_transform(df[['MONTH_START']]).toarray(),
    columns=one_encoder_MONTH_START.get_feature_names())
    df_SB = pd.DataFrame(one_encoder_SUB_BASIN.fit_transform(df[['SUB BASIN']]).toarray(),
    columns=one_encoder_SUB_BASIN.get_feature_names())
    df = pd.concat([df_GC, df_NA, df_MS, df_SB, df],axis=1)
    df.drop(columns=['NATURE', 'GENERAL_CATEGORY', 'MONTH_START', 'SUB BASIN'], axis=1,
    inplace= True)
    # df = pd.get_dummies(df, columns=['GENERAL_CATEGORY'])
    # df = pd.get_dummies(df, columns=['NATURE'])
    # df = pd.get_dummies(df, columns=['MONTH_START'])
    # df = pd.get_dummies(df, columns=['SUB BASIN'])

    df.drop(columns=[ 'ISO', 'YEAR'],axis=1,inplace=True)
    df = df.dropna()
    features = [x for x in df.columns.values if x != 'TOTAL_AFFECTED']
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df['TOTAL_AFFECTED'], test_size=0.2, random_state=42)
    # print('X_train_info')
    # print(X_train.info())
    model_fit_save(X_train, y_train, X_test, y_test)

def model_fit_save(train_x, train_y, test_x, test_y):

    ## Training the model
    r1 = LinearRegression()
    #r2 = RandomForestRegressor(n_estimators=10, random_state=1)
    r3 =   SVR(kernel = 'rbf')

    er = VotingRegressor([('lr', r1), 
                          #('rf', r2),
                          ('svr_rbf', r3)])

    er.fit(train_x, train_y)

    ### Evaluating based on the train data
    y_pred = er.predict(test_x)
    print('Mean Absolute Error:', mean_absolute_error(test_y, y_pred))  
    print('Mean Squared Error:', mean_squared_error(test_y, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(mean_squared_error(test_y, y_pred)))


    ## Saving the model
    # Save the model as a pickle in a file 
    joblib.dump(er, 'model.pkl') 

########### 2. Inference of the model #############

def predict_output(X, pickleFile=None):
    if pickleFile == None:
        pickleFile = "./model.pkl"
    # Load the model from the file 
    model = joblib.load(pickleFile)  
  
    # Use the loaded model to make predictions 
    print(X.info())
    y_pred = model.predict(X)
    return y_pred.copy()

########### 3. StreamLit UI and processing the input ###############
@st.cache(persist=True, allow_output_mutation=True, suppress_st_warning=True)
def load_data(data_url):
    data = pd.read_csv(data_url)
    data['ISO_TIME'] = pd.to_datetime(data['ISO_TIME'])
   
    #data = data.dropna()
    data['SID']=1
    data['USA_R34_m'] = data['R34'].fillna(data['R34'].mean())
    data['USA_R50_m'] = data['R50'].fillna(data['R50'].mean())
    data['USA_R64_m'] = data['R64'].fillna(data['R64'].mean())
    data['USA_R34_m'] = data['USA_R34_m']*2000
    data['USA_R50_m'] = data['USA_R50_m']*2000
    data['USA_R64_m'] = data['USA_R64_m']*2000
    lowercase = lambda x: str(x).lower()
   # data.dropna(subset=['LAT', 'LON', 'USA_R34_NE'], inplace=True)
    
    data.rename(columns={'LAT': 'lat'},inplace=True)
    data.rename(columns={'LON': 'lon'},inplace=True)
    data.rename(lowercase, axis='columns', inplace=True)
    return data



# utilities for cleaning the data
def agg_coords(series):
    coords_list = series.tolist()
    return coords_list


def get_iso(coords):
    return rg.search(coords, mode = 1)[0]["cc"]


def dist_(x1,y1,x2,y2):
    try:
        x = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    except:
        x = 0
    return x
        
def tot_dist(lst):
    dist_lst=[]
    lat = []
    lon=[]
    for i in range(0, len(lst)):
        lat.append( float(lst[i][0]) )
    for i in range(0, len(lst)):
        lon.append( float(lst[i][0]) )
  
    for i, (a, b) in enumerate(zip(lat, lon)):
        
        try:
            
            x1 = a
            x2 = lat[i+1]
            y1 = b
            y2 = lon[i+1]
         
            dist_lst.append(dist_(float(x1),float(y1),float(x2),float(y2)))
            
        except:
            pass

    try:
        avg_dist = sum(dist_lst)/len(dist_lst)
    except:
        avg_dist = 0
    return sum(dist_lst)


def get_distance(coords_list):
    dist_list = []
    n = len(coords_list)
    if n==0:
        return 0
    else:
        for i in range(n):
            if i>0:
                dist_list.append(distance.distance(coords_list[i-1], coords_list[i]).km)
        return sum(dist_list)


def max_min(x):
        diff =  x.max() - x.min()
        days, seconds = diff.days, diff.seconds
        hours = days * 24 + seconds // 3600
        return hours


if __name__ == "__main__":
    ## Check for the model.pkl file if not train the data
    # global one_encoder_GENERAL_CATEGORY, one_encoder_MONTH_START, one_encoder_NATURE, one_encoder_SUB_BASIN
    one_encoder_GENERAL_CATEGORY = OneHotEncoder()
    one_encoder_NATURE = OneHotEncoder()
    one_encoder_MONTH_START = OneHotEncoder()
    one_encoder_SUB_BASIN = OneHotEncoder()
    input_file = 'OUTPUT_WBI_exposer_cyclones_v11.xlsx'
    init_data = pd.read_excel(input_file)
    if os.path.exists('./model.pkl'):
        pass
        # one_encoder_GENERAL_CATEGORY.fit(init_data[['GENERAL_CATEGORY']])
        # one_encoder_MONTH_START.fit(init_data[['MONTH_START']])
        # one_encoder_NATURE.fit(init_data[['NATURE']])
        # one_encoder_SUB_BASIN.fit(init_data[['SUB BASIN']])
    else:
        preprocessing(input_file)
    one_encoder_GENERAL_CATEGORY.fit(init_data[['GENERAL_CATEGORY']])
    one_encoder_MONTH_START.fit(init_data[['MONTH_START']])
    one_encoder_NATURE.fit(init_data[['NATURE']])
    one_encoder_SUB_BASIN.fit(init_data[['SUB BASIN']])
    # what day is today?
    dt = datetime.datetime.today()
    #what month?
    m =  dt.month
    #sidebar text asking for upload
    st.sidebar.title("Upload")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    st.sidebar.title("Upload tracks to calculate exposer exposer")
    st.sidebar.markdown("Upload file with coordinates and radii of cyclone, coordinates and radii of 34, 50, 64 knots per hour, NAMES: LAT, LON, R34, R50, R50")
    #default upload for now
    if uploaded_file is not None:
        data1 = load_data(uploaded_file)
    else:
        data1 = load_data('is.csv')
    data = data1.copy()


    ## Midpoint calculation and visualization
    midpoint = (np.average(data['lat']), np.average(data['lon']))
    lst_lat, lst_lon = data["lat"].tolist(),data["lon"].tolist()
    st.title("Visualization of Cyclone path")
    
    data["COORDS"] = data[["lat", "lon"]].values.tolist()
    data["COORDS"] = data['COORDS'].apply(lambda x: (x[0], x[1]))
    data["ISO"] = data['COORDS'].apply(get_iso)

 
    paths = data.groupby(["ISO"]).agg({"COORDS": agg_coords, 'iso_time': max_min, 'lon': 'first', 'lat':'first'}).reset_index().rename(columns = {"ISO_TIME": "TOTAL_HOURS_IN_LAND", "lon":"COORD_1", "lat":"COORD_2"})
    
    paths["DISTANCE_TRACK_VINCENTY"] = paths.COORDS.apply(tot_dist)

    wbi = pd.read_csv('wbi.csv')
    paths =  pd.merge(paths, wbi, how= 'left', left_on='ISO', right_on = 'Country Code')

    #ask for more features
    MONTH_START = st.sidebar.text_input("MONTH_START", m)

    GENERAL_CATEGORY = st.sidebar.selectbox("Select GENERAL_CATEGORY", init_data["GENERAL_CATEGORY"].unique())

    MAX_WIND     = st.sidebar.text_input("MAX_WIND", max(init_data["MAX_WIND"].unique()))
    MIN_PRES     = st.sidebar.text_input("MIN_PRES", max(init_data["MIN_PRES"].unique()))
    SUB_BASIN     = st.sidebar.text_input("SUB BASIN", max(init_data["SUB BASIN"].unique()))
    #MIN_DIST2LAND    = st.sidebar.text_input("MIN_DIST2LAND")
    MAX_STORMSPEED   = st.sidebar.text_input("MAX_STORMSPEED", max(init_data["MAX_STORMSPEED"].unique()))
    #TOTAL_HOURS_EVENT = st.sidebar.text_input("TOTAL_HOURS_EVENT")
    #TOTAL_HOURS_IN_LAND = st.sidebar.text_input("TOTAL_HOURS_IN_LAND")
    NATURE = st.sidebar.text_input("NATURE", 'TS')
    # GENERAL_CATEGORY = st.sidebar.text_input("GENERAL_CATEGORY",5)
    paths['MONTH_START'] = int(MONTH_START)
    paths['MIN_PRES'] = int(MIN_PRES)
    paths['SUB_BASIN'] = str(SUB_BASIN)
    paths['MAX_STORMSPEED'] = float(MAX_STORMSPEED)
    paths['NATURE'] = NATURE
    paths['GENERAL_CATEGORY'] = GENERAL_CATEGORY
    paths['MAX_WIND'] = MAX_WIND
    # print(f'{GENERAL_CATEGORY} - {NATURE} - {SUB_BASIN} - {MONTH_START} - {one_encoder_NATURE.get_feature_names()}')


    ######### Preprocessing the data for model ############
    df_gc = pd.DataFrame(one_encoder_GENERAL_CATEGORY.transform(paths[['GENERAL_CATEGORY']]).toarray(),
    columns=one_encoder_GENERAL_CATEGORY.get_feature_names())
    df_ms = pd.DataFrame(one_encoder_MONTH_START.transform(paths[['MONTH_START']]).toarray(),
    columns=one_encoder_MONTH_START.get_feature_names())
    df_na = pd.DataFrame(one_encoder_NATURE.transform(paths[['NATURE']]).toarray(),
    columns=one_encoder_NATURE.get_feature_names())
    df_sb = pd.DataFrame(one_encoder_SUB_BASIN.transform(paths[['SUB_BASIN']]).toarray(),
    columns=one_encoder_SUB_BASIN.get_feature_names())
    paths = pd.concat([df_gc,df_ms,df_na,df_sb,paths],axis=1)

    paths.drop(columns=['NATURE', 'SUB_BASIN', 'GENERAL_CATEGORY', 'MONTH_START',
    'ISO', 'COORDS','iso_time'], axis=1, inplace=True)
    # paths.to_csv('./test1.csv')

    paths = paths.fillna(0.0)

    st.write(paths)


    if st.checkbox("Show Raw data", False):
        st.subheader('Raw Data')
        st.write(paths)
        st.write(data)
    if st.button("Generate Score for metrics model"):
        output = predict_output(paths)
        st.write(f'The predicted affected population is {output}')
        # st.write(f'Mean absolute error of the population is {}')
        st.balloons()


########## training input #########
# x0_Cat 1                   711 non-null float64
# x0_Cat 2                   711 non-null float64
# x0_Cat 3                   711 non-null float64
# x0_Cat 4                   711 non-null float64
# x0_Cat 5                   711 non-null float64
# x0_TD                      711 non-null float64
# x0_TS                      711 non-null float64
# x0_TS                      711 non-null float64
# x0_DS                      711 non-null float64
# x0_ET                      711 non-null float64
# x0_MX                      711 non-null float64
# x0_NR                      711 non-null float64
# x0_SS                      711 non-null float64
# x0_TS                      711 non-null float64
# x0_TS                      711 non-null float64
# x0_1                       711 non-null float64
# x0_2                       711 non-null float64
# x0_3                       711 non-null float64
# x0_4                       711 non-null float64
# x0_5                       711 non-null float64
# x0_6                       711 non-null float64
# x0_7                       711 non-null float64
# x0_8                       711 non-null float64
# x0_9                       711 non-null float64
# x0_10                      711 non-null float64
# x0_11                      711 non-null float64
# x0_12                      711 non-null float64
# x0_AS                      711 non-null float64
# x0_BB                      711 non-null float64
# x0_CP                      711 non-null float64
# x0_CS                      711 non-null float64
# x0_EA                      711 non-null float64
# x0_EP                      711 non-null float64
# x0_GM                      711 non-null float64
# x0_NAm                     711 non-null float64
# x0_SI                      711 non-null float64
# x0_SP                      711 non-null float64
# x0_WA                      711 non-null float64
# x0_WP                      711 non-null float64
# TOTAL_HOURS_IN_LAND        711 non-null float64
# MAX_WIND                   711 non-null float64
# MIN_PRES                   711 non-null float64
# MAX_STORMSPEED             711 non-null float64
# DISTANCE_TRACK_VINCENTY    711 non-null float64
# POP_MAX_34_ADJ             711 non-null float64
# POP_MAX_50_ADJ             711 non-null float64
# POP_MAX_64_ADJ             711 non-null float64
# POP_DEN_SQ_KM              711 non-null float64
# RURAL_POP(%)               711 non-null float64
# POP_TOTAL                  711 non-null float64
# RURAL_POP                  711 non-null float64
# HDI                        711 non-null float64
# COORDS_1                   711 non-null float64
# COORDS_2                   711 non-null float64

##### prediction input #########
# x0_Cat 1                   18 non-null float64
# x0_Cat 2                   18 non-null float64
# x0_Cat 3                   18 non-null float64
# x0_Cat 4                   18 non-null float64
# x0_Cat 5                   18 non-null float64
# x0_TD                      18 non-null float64
# x0_TS                      18 non-null float64
# x0_1                       18 non-null float64
# x0_2                       18 non-null float64
# x0_3                       18 non-null float64
# x0_4                       18 non-null float64
# x0_5                       18 non-null float64
# x0_6                       18 non-null float64
# x0_7                       18 non-null float64
# x0_8                       18 non-null float64
# x0_9                       18 non-null float64
# x0_10                      18 non-null float64
# x0_11                      18 non-null float64
# x0_12                      18 non-null float64
# x0_DS                      18 non-null float64
# x0_ET                      18 non-null float64
# x0_MX                      18 non-null float64
# x0_NR                      18 non-null float64
# x0_SS                      18 non-null float64
# x0_TS                      18 non-null float64
# x0_AS                      18 non-null float64
# x0_BB                      18 non-null float64
# x0_CP                      18 non-null float64
# x0_CS                      18 non-null float64
# x0_EA                      18 non-null float64
# x0_EP                      18 non-null float64
# x0_GM                      18 non-null float64
# x0_NAm                     18 non-null float64
# x0_SI                      18 non-null float64
# x0_SP                      18 non-null float64
# x0_WA                      18 non-null float64
# x0_WP                      18 non-null float64
# DISTANCE_TRACK_VINCENTY    18 non-null float64
# MIN_PRES                   18 non-null int64
# MAX_STORMSPEED             18 non-null float64




#### Difference
# TOTAL_HOURS_IN_LAND        711 non-null float64
# MAX_WIND                   711 non-null float64
# POP_MAX_34_ADJ             711 non-null float64
# POP_MAX_50_ADJ             711 non-null float64
# POP_MAX_64_ADJ             711 non-null float64
# POP_DEN_SQ_KM              711 non-null float64
# RURAL_POP(%)               711 non-null float64
# POP_TOTAL                  711 non-null float64
# RURAL_POP                  711 non-null float64
# HDI                        711 non-null float64
# COORDS_1                   711 non-null float64
# COORDS_2                   711 non-null float64