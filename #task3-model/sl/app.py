import streamlit  as st

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
# ??? Perform min_max_scaler to YEAR, TOTAL_HRS,MONTH
#from sklearn.preprocessing import MinMaxScaler, StandardScaler
import reverse_geocoder as rg

import math 
import pandas as pd
import numpy as np
import pydeck as pdk
#preprocessing
import geopy
import pickle
import datetime
from geopy import distance
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

initial_df = pd.read_excel('OUTPUT_WBI_exposer_cyclones_v11.xlsx')

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
DATA_URL = ("is.csv")

@st.cache(persist=True, allow_output_mutation=True, suppress_st_warning=True)

#loading clients track
def load_data(data_url):
    data = pd.read_csv(DATA_URL)
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

#if file uploaded : viz for cyclone, ask for more data in text input boxes, ssow data
if uploaded_file is not None:
    data1 = load_data(uploaded_file)
    #st.write(data.head())
    #paths = data.groupby(["ISO"]).agg({"COORDS": agg_coords}).reset_index()
    #paths["VINCENTY_LENGTH"] = paths.COORDS.apply(lambda x: distance.distance(x[0], x[-1]).km)
    data = data1.copy()
    


    #DISTANCE_TRACK_VINCENTY = st.sidebar.text_input("DISTANCE_TRACK_VINCENTY", )



    #data = data.dropna()




    midpoint = (np.average(data['lat']), np.average(data['lon']))
    lst_lat, lst_lon = data["lat"].tolist(),data["lon"].tolist()
    
    st.title("Visualization of Cyclone path")
    

    layer_34 = pdk.Layer(

            "ScatterplotLayer",
            data=data[['usa_r34_m', 'lat', 'lon']],
            pickable=True,
                    opacity=0.008,
                    stroked=True,
                    filled=True,
                    
                    line_width_min_pixels=0,
                    get_position=["lon", "lat"],
                    get_radius="usa_r34_m",
                    get_fill_color=[255, 140, 0],
                    get_line_color=[0, 0, 0],)



    layer_64 = pdk.Layer(

            "ScatterplotLayer",
            data=data[['usa_r64_m', 'lat', 'lon']],
            pickable=True,
                    opacity=0.008,
                    stroked=True,
                    filled=True,
                    
                    line_width_min_pixels=0,
                    get_position=["lon", "lat"],
                    get_radius="usa_r64_m",
                    get_fill_color=[255, 140, 56],
                    get_line_color=[0, 0, 0],)


    layer_50 = pdk.Layer(

            "ScatterplotLayer",
            data=data[['usa_r50_m', 'lat', 'lon']],
            pickable=True,
                    opacity=0.008,
                    stroked=True,
                    filled=True,
                    
                    line_width_min_pixels=0,
                    get_position=["lon", "lat"],
                    get_radius="usa_r50_m",
                    get_fill_color=[255, 80, 0],
                    get_line_color=[0, 0, 0],)


    st.write(pdk.Deck(
        map_style = "mapbox://styles/mapbox/light-v9",
        initial_view_state = {
        "latitude": midpoint[0],
        "longitude": midpoint[1],
        "zoom": 6,
        "pitch": 50

        },
        layers=[
            
            layer_34,
            layer_50,
            layer_64

        ]
    ))

    data["COORDS"] = data[["lat", "lon"]].values.tolist()
    data["COORDS"] = data['COORDS'].apply(lambda x: (x[0], x[1]))
    data["ISO"] = data['COORDS'].apply(get_iso)
    
    data["SID"] = 1
    def max_min(x):
        diff =  x.max() - x.min()
        days, seconds = diff.days, diff.seconds
        hours = days * 24 + seconds // 3600
        return hours
    paths = data.groupby(["SID", "ISO"]).agg({"COORDS": agg_coords, 'iso_time': max_min, 'lon': 'first', 'lat':'first'}).reset_index().rename(columns = {"ISO_TIME": "TOTAL_HOURS_IN_LAND", "lon":"COORD_1", "lat":"COORD_2"})
    def max_min(x):
        return x.max() - x.min()
    
   
    paths["DISTANCE_TRACK_VINCENTY"] = paths.COORDS.apply(tot_dist)



    #ask for more features
    MONTH_START = st.sidebar.text_input("MONTH_START", m)

    GENERAL_CATEGORY = st.sidebar.selectbox("Select GENERAL_CATEGORY", initial_df["GENERAL_CATEGORY"].unique(), 'Cat 4')

    MAX_WIND     = st.sidebar.text_input("MAX_WIND", max(initial_df["MAX_WIND"].unique()))
    MIN_PRES     = st.sidebar.text_input("MIN_PRES", max(initial_df["MIN_PRES"].unique()))
    SUB_BASIN     = st.sidebar.text_input("SUB BASIN", max(initial_df["SUB BASIN"].unique()))
    #MIN_DIST2LAND    = st.sidebar.text_input("MIN_DIST2LAND")
    MAX_STORMSPEED   = st.sidebar.text_input("MAX_STORMSPEED", max(initial_df["MAX_STORMSPEED"].unique()))
    #OTAL_HOURS_EVENT = st.sidebar.text_input("TOTAL_HOURS_EVENT")
    #TOTAL_HOURS_IN_LAND = st.sidebar.text_input("TOTAL_HOURS_IN_LAND")
    NATURE = st.sidebar.text_input("NATURE", 5)
    GENERAL_CATEGORY = st.sidebar.text_input("GENERAL_CATEGORY",5)
    paths['MONTH_START'] = MONTH_START
    paths['MIN_PRES'] = MIN_PRES
    paths['SUB_BASIN'] = SUB_BASIN
    paths['MAX_STORMSPEED'] = MAX_STORMSPEED
    paths['NATURE'] = NATURE


    if st.checkbox("Show Raw data", False):
        st.subheader('Raw Data')
        st.write(paths)
        st.write(data)

# load the model from disk
#loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
import matplotlib.pyplot as plt

#st.write(initial_df.head())
initial_df = initial_df.drop_duplicates(subset=['NAME','ISO','YEAR'], keep="last")
initial_df['COORDS_both']=initial_df['COORDS'].str[2:18]
initial_df['COORDS_1'] = initial_df['COORDS_both'].str.split(", ", n = 1, expand = True)[0]
initial_df['COORDS_2'] = initial_df['COORDS_both'].str.split(", ",  expand = True)[1]
initial_df['COORDS_2'] = initial_df['COORDS_2'].str.split(")",  expand = True)[0]
initial_df['COORDS_1'] = initial_df['COORDS_1'].astype(float)
initial_df['COORDS_2'] = initial_df['COORDS_2'].astype(float)

initial_df['POP_MAX_50_ADJ'] = initial_df['POP_MAX_50_ADJ']-initial_df['POP_MAX_34_ADJ']
initial_df['POP_MAX_50'] = initial_df['POP_MAX_50']-initial_df['POP_MAX_34']
initial_df['POP_MAX_34_ADJ'] = initial_df['POP_MAX_34_ADJ']-initial_df['POP_MAX_50_ADJ']
initial_df['POP_MAX_34'] = initial_df['POP_MAX_34']-initial_df['POP_MAX_50']
initial_df['POP_MAX_50'][initial_df['POP_MAX_50'] < 0] = 0
initial_df['POP_MAX_50_ADJ'][initial_df['POP_MAX_50_ADJ'] < 0] = 0
initial_df['POP_MAX_64'][initial_df['POP_MAX_64'] < 0] = 0
initial_df['POP_MAX_64_ADJ'][initial_df['POP_MAX_64_ADJ'] < 0] = 0
req_columns = [ 'SUB BASIN', 'ISO', 'YEAR', 'MONTH_START',
       #'MONTH_END', droppping high correlation with with month start
       # 'DATE_START', 'DATE_END', 'DATE_LAND_START',
         
       'TOTAL_HOURS_IN_LAND', 'NATURE',
       'GENERAL_CATEGORY', 'MAX_WIND', 'MIN_PRES', #'MIN_DIST2LAND',
       'MAX_STORMSPEED',

        #'MAX_USA_SSHS', 
       #'MAX_USA_SSHS_INLAND', 
       #'V_LAND_KN',
       #'COORDS_MAX_WINDS', 'COORDS_MIN_DIST2LAND', could not recreate
       #'DISTANCE_TRACK',
       'DISTANCE_TRACK_VINCENTY', 
       #ffejfkukuhftvnruncnvvegfjhhvdfjd'34KN_POP', '64KN_POP', '96KN_POP',
       'POP_MAX_34_ADJ', 'POP_MAX_50_ADJ', 'POP_MAX_64_ADJ', 
       # '64KN_ASSETS', '34KN_ASSETS', '96KN_ASSETS',
       'POP_DEN_SQ_KM', 'RURAL_POP(%)', 'POP_TOTAL', 'RURAL_POP', 
       'HDI',
       #'TOTAL_DAMAGE_(000$)', 'TOTAL_DEATHS', 
       'TOTAL_AFFECTED', 
       #'dist',
       # 'COORDS_both',
         'COORDS_1', 'COORDS_2']
df = initial_df[req_columns]
df = pd.get_dummies(df, columns=['GENERAL_CATEGORY'])
df = pd.get_dummies(df, columns=['NATURE'])
df = pd.get_dummies(df, columns=['MONTH_START'])
df = pd.get_dummies(df, columns=['SUB BASIN'])

df.drop(columns=[#'month',
                  # 'Income_level_Final', 
                   'ISO', 
                   'YEAR'
                   ],axis=1,inplace=True)
df = df.dropna()
## Split df

features = [x for x in df.columns.values if x != 'TOTAL_AFFECTED']
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df['TOTAL_AFFECTED'], test_size=0.2, random_state=42)





from sklearn.linear_model import LinearRegression

from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import RidgeCV

select_model = st.selectbox("Select model",["ensembled","svr","nn"])

def esmld():
    r1 = LinearRegression()
    #r2 = RandomForestRegressor(n_estimators=10, random_state=1)
    r3 =   SVR(kernel = 'rbf')

    er = VotingRegressor([('lr', r1), 
                          #('rf', r2),
                          ('svr_rbf', r3)])



    er.fit(X_train, y_train)
    y_pred = er.fit(X_train, y_train).predict(X_test)
    st.write('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))  
    st.write('Mean Squared Error:', mean_squared_error(y_test, y_pred))  
    st.write('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))


    print(er.fit(X_train, y_train).predict(X_test))

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

