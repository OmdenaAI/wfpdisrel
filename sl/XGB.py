
from helper import *

import pickle
import datetime

import pandas as pd


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import xgboost as xgb


model = pickle.load(open('XGB.model', 'rb'))
kmeans = pickle.load(open('kmeans.cluster', 'rb'))


# what day is today?
dt = datetime.datetime.today()
#what month?
m =  dt.month


def preprocessing(inputFile=None):

    if inputFile:

        data = pd.read_csv(inputFile)

        data['ISO_TIME'] = pd.to_datetime(data['ISO_TIME'])
        data['YEAR'] = data['ISO_TIME'].dt.year
    
        data['SID'] = 1

        lowercase = lambda x: str(x).lower()
        
        data.rename(columns={'LAT': 'lat'},inplace=True)
        data.rename(columns={'LON': 'lon'},inplace=True)
        data.rename(lowercase, axis='columns', inplace=True)

        data["COORDS"] = data[["lat", "lon"]].values.tolist()
        data["COORDS"] = data['COORDS'].apply(lambda x: (x[0], x[1]))
        data["ISO"] = data['COORDS'].apply(get_iso)

        paths = data.copy()

        ##################################
        ### Artificially create features##
        ##################################

        paths['MONTH_START'] = 10
        paths['MIN_PRES'] = 20
        paths['BASIN'] = '10'

        paths['SUB_BASIN'] = '20'
        paths['MAX_STORMSPEED'] = 30
        paths['NATURE'] = '20'
        paths['MAX_WIND'] = 20
        paths['MIN_PRES'] = 40

        paths['TOTAL_HOURS_EVENT'] = 12
        paths['TOTAL_HOURS_IN_LAND'] = 34

        paths['MIN_DIST2LAND'] = 11


        paths['GENERAL_CATEGORY'] = '23'

        ## Temp values for pop_max
        paths['POP_MAX_34_ADJ'] = 100
        paths['POP_MAX_50_ADJ'] = 120
        paths['POP_MAX_64_ADJ'] = 130

        paths['MAX_SSH_7'] = 1
        paths['MAX_SSH_SS'] = 0

        paths['MAX_USA_SSHS'] = 0
        paths['MAX_USA_SSHS_INLAND'] = '0'
        paths['V_LAND_KN'] = 1

        ############
        ##############
        #################

        ## Merge WBI
        wbi = pd.read_csv("result-wbi.csv")
        df = pd.merge(paths, wbi, left_on=["ISO", "season"], right_on=['Alpha-2 code', 'Year'], how='left')


    ## At this point. Inf file & original df same

    else:
        df = pd.read_csv('OUTPUT_WBI_exposer_cyclones_v14.csv', sep=";")
        df.rename(columns={'SUB BASIN': 'SUB_BASIN'},inplace=True)
        df.rename(columns={'RURAL_POP(%)': 'RURAL_POP'},inplace=True)
        df.rename(columns={'YEAR': 'season'},inplace=True)
        



    df["SUB_BASIN"]= df["SUB_BASIN"].replace('MM', np.nan) 
    df["BASIN"]= df["BASIN"].replace('MM', np.nan)
    df['SUB_BASIN']= np.where(df['SUB_BASIN'].isnull(), df['BASIN'], df['SUB_BASIN'])
    df['MAX_USA_SSHS_INLAND'] = df['MAX_USA_SSHS_INLAND'].astype('object')

    MAX_SSHS_dict = {-7: 218, -6: 1, -5: 2, -4: 8, -3: 7, -2: 2, 
                     -1: 97, 0: 272, 1: 161, 2: 91, 3: 80, 4: 43, 5: 9}
    df['SSH_Freq'] = df['MAX_USA_SSHS_INLAND'].map(MAX_SSHS_dict)


    ## load preprocessing modules
    median_imputer = pickle.load(open('median.imp', 'rb'))
    OHenc = pickle.load(open('enc.encoder', 'rb'))
    scale = pickle.load(open('scale.scaler', 'rb'))

    df[median_cols] = median_imputer.transform(df[median_cols])

    new_cats(df)
    new_cont(df)


    dummies = pd.DataFrame(data=OHenc.transform(df[object_cols]), columns=dummy_cols)

    df = pd.concat([df, dummies], axis=1)


    df = df.fillna(100)
    X = df[features]

    X = scale.transform(X)

    km_pred = kmeans.predict(X).reshape(-1, 1)
    X = np.column_stack((X, km_pred))

    ## If inference file - return
    # if not, pass to train_model
    if inputFile:
        return(X)
    else:
        df['TOTAL_AFFECTED'] = np.log(df['TOTAL_AFFECTED'] + 1)
        X_train, X_test, y_train, y_test = train_test_split(X, df['TOTAL_AFFECTED'], test_size=0.2, random_state=42)
        train_model(X_train, y_train, X_test, y_test)





def train_model(train_x, train_y, test_x, test_y):

    xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.2, learning_rate = 0.1,
                max_depth = 4, alpha = 10, n_estimators = 1000)

    xg_reg.fit(train_x, train_y)

    predtest = xg_reg.predict(test_x)**2

    print('mean absolute error : ', mean_absolute_error(test_y**2, predtest))
    print('mean squared error  : ', mean_squared_error(test_y**2, predtest))
    print('root mean squared error : ', np.sqrt(mean_squared_error(test_y**2, predtest)))
                    
    pickle.dump(xg_reg, open('XGB.model', 'wb'))


def infer(df):

    # Load ML Model
    pred = model.predict(df)

    return(pred)