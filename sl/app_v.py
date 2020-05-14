## Importing the required libraries
import pandas as pd
import numpy as np
import pickle
import datetime
import os
import reverse_geocoder as rg
import math
from geopy import distance

# Load Model Libraries
from Custom_SVR.SVR import My_SVR_Model
from Custom_GBM.GBM import My_GBM_Model
from Custom_XGB.XGB import My_XGB_Model
from Custom_NN.NN import My_NN_Model

import json

### stream Lit UI
import streamlit as st
import pydeck as pdk


@st.cache(persist=True, allow_output_mutation=True, suppress_st_warning=True)
def load_data(data_url):
    data = pd.read_csv(data_url)
    data['ISO_TIME'] = pd.to_datetime(data['ISO_TIME'])
   
    #data = data.dropna()
    #data['SID']=1
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
    

    #sidebar text asking for upload
    st.sidebar.title("Upload")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    st.sidebar.title("Upload tracks to calculate exposer exposer")
    st.sidebar.markdown("Upload file with coordinates and radii of cyclone, coordinates and radii of 34, 50, 64 knots per hour, NAMES: LAT, LON, R34, R50, R64")
    if uploaded_file is not None:
        data1 = load_data(uploaded_file)
        init_data = pd.read_csv('./OUTPUT_WBI_exposer_cyclones_v14.csv',sep=';')
        # what day is today?
        current_date = datetime.datetime.today()
        #what month?
        month =  current_date.month
        data = data1.copy()
        ## Midpoint calculation and visualization
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
        paths = data.groupby(["SID", "ISO", "season"]).agg({"COORDS": agg_coords, 'iso_time': max_min}).reset_index().rename(columns = {"iso_time": "TOTAL_HOURS_IN_LAND"})
        countries = paths['ISO'].copy()
        paths["DISTANCE_TRACK_VINCENTY"] = paths.COORDS.apply(tot_dist)

        #ask for more features
        MONTH_END = st.sidebar.text_input("MONTH_START", month)

        GENERAL_CATEGORY = st.sidebar.selectbox("Select GENERAL_CATEGORY", np.sort(init_data["GENERAL_CATEGORY"].unique()))

        MAX_WIND     = st.sidebar.text_input("MAX_WIND", max(init_data["MAX_WIND"].unique()))
        MIN_PRES     = st.sidebar.text_input("MIN_PRES", max(init_data["MIN_PRES"].unique()))
        BASIN = st.sidebar.selectbox("BASIN", init_data["BASIN"].unique())
        SUB_BASIN     = st.sidebar.text_input("SUB BASIN", max(init_data["SUB BASIN"].unique()))
        MAX_STORMSPEED   = st.sidebar.text_input("MAX_STORMSPEED", max(init_data["MAX_STORMSPEED"].unique()))

        V_LAND_KN = st.sidebar.text_input("V_LAND", max(init_data['V_LAND_KN']))

        MAX_USA_SSHS_INLAND = st.sidebar.selectbox("SSHS INLAND", np.sort(init_data["MAX_USA_SSHS_INLAND"].unique()))
        MAX_USA_SSHS = st.sidebar.selectbox("SSHS", np.sort(init_data["MAX_USA_SSHS"].unique()))
        

        NATURE = st.sidebar.text_input("NATURE", 'TS')
        paths['MONTH_END'] = int(MONTH_END)
        paths['MONTH_START'] = int(MONTH_END)
        paths['MIN_PRES'] = int(MIN_PRES)
        paths['BASIN'] = str(BASIN)
        paths['SUB BASIN'] = str(SUB_BASIN)
        paths['MAX_STORMSPEED'] = float(MAX_STORMSPEED)
        paths['NATURE'] = NATURE
        paths['GENERAL_CATEGORY'] = str(GENERAL_CATEGORY)
        paths['MAX_WIND'] = 116
        paths['MAX_USA_SSHS_INLAND'] = MAX_USA_SSHS_INLAND
        paths['MAX_USA_SSHS'] = MAX_USA_SSHS


        
        #paths['RURAL_POP'] = 1000000
        #paths['POP_TOTAL'] = 100000000
        paths['RURAL_POP(%)'] = 70
        paths['V_LAND_KN'] = 100
        paths['MIN_DIST2LAND'] = 0
        
        #paths['GDP per capita (constant 2010 US$)'] = 8607.657082
        #paths["Adjusted savings: education expenditure (% of GNI)"] = 2.8678781
        #paths['Net flows from UN agencies US$'] = 0
        #paths['HDI'] = 0.816

        if st.checkbox("Show Raw data", False):
            st.subheader('Raw Data')
            st.write(paths)
            st.write(data)
        algorithm = st.selectbox("Select the algorithm", ['SVR', 'GBM','XGB', "NN"])

        if algorithm == 'SVR':
            paths['GDP per capita (constant 2010 US$)'] = 8607.657082
            paths["Adjusted savings: education expenditure (% of GNI)"] = 2.8678781
            paths['Net flows from UN agencies US$'] = 0
            paths['HDI'] = 0.816

        if os.path.exists(f'./{algorithm.lower()}-data.json'):
            with open(f'./{algorithm.lower()}-data.json','r') as datafile:
                mae = json.load(datafile)
                mae = mae[algorithm.lower()]
                mae = mae['mae']
                st.write(f'{mae} is the mean absolute error')
            pass

        if st.button("Generate Score for metrics model"):

            models = {
                'SVR': My_SVR_Model(init_data), 
                'GBM': My_GBM_Model(init_data),
                'XGB': My_GBM_Model(init_data), 
                'NN': My_NN_Model(init_data)
            }

            model = models[algorithm]

            #output = np.round(model.infer(paths))
            output = model.infer(paths)
            st.write(pd.concat([countries,pd.Series(output,name='Population Affected')],axis=1))
            st.balloons()