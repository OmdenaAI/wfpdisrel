
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



class My_XGB_Model:
    def __init__(self, inputfile=None):

        self.model = pickle.load(open('XGB.model', 'rb'))
        self.kmeans = pickle.load(open('kmeans.cluster', 'rb'))

        self.init_data = inputfile.copy()

        ## load preprocessing modules
        self.median_imputer = pickle.load(open('median.imp', 'rb'))
        self.OHenc = pickle.load(open('enc.encoder', 'rb'))
        self.scale = pickle.load(open('scale.scaler', 'rb'))


    def preprocessing(self, data):


        #data = self.init_data

        paths = data.copy()

        ## Merge WBI
        wbi = pd.read_csv("result-wbi.csv")
        ivan = pd.read_csv("IVAN_OUTPUT.csv")
        total_pop = pd.read_csv('total_pop.csv')
        iso_code = pd.read_csv('ISO-codes.csv')[['Country', 'Alpha-2 code']]

        ivan = pd.merge(ivan, iso_code, left_on='COUNTRY', right_on='Country', how='inner')
        
        df = pd.merge(paths, wbi, left_on=["ISO", "season"], right_on=['Alpha-2 code', 'Year'], how='left')
        df = pd.merge(df, ivan, left_on=['ISO', "season"], right_on=['Alpha-2 code', 'SEASON'], how='left')
        df = pd.merge(df, total_pop, left_on='ISO', right_on='Country Code', how='left')


        df.rename(columns={'SUB BASIN': 'SUB_BASIN'},inplace=True)

        df['POP_MAX_34_ADJ'] = df['pop_max_34']*df['2018']/df['2015']
        df['POP_MAX_50_ADJ'] = df['pop_max_50']*df['2018']/df['2015']
        df['POP_MAX_64_ADJ'] = df['pop_max_64']*df['2018']/df['2015']

        df["SUB_BASIN"]= df["SUB_BASIN"].replace('MM', np.nan) 
        df["BASIN"]= df["BASIN"].replace('MM', np.nan)
        df['SUB_BASIN']= np.where(df['SUB_BASIN'].isnull(), df['BASIN'], df['SUB_BASIN'])

        df['MAX_USA_SSHS_INLAND'] = df['MAX_USA_SSHS_INLAND'].astype('object')

        MAX_SSHS_dict = {-7: 218, -6: 1, -5: 2, -4: 8, -3: 7, -2: 2, 
                        -1: 97, 0: 272, 1: 161, 2: 91, 3: 80, 4: 43, 5: 9}
        df['SSH_Freq'] = df['MAX_USA_SSHS_INLAND'].map(MAX_SSHS_dict)

        df[median_cols] = self.median_imputer.transform(df[median_cols])

        new_cats(df)
        new_cont(df)


        dummies = pd.DataFrame(data=self.OHenc.transform(df[object_cols]), columns=dummy_cols)

        df = pd.concat([df, dummies], axis=1)


        df = df.fillna(100)
        X = df[features]

        X = self.scale.transform(X)

        km_pred = self.kmeans.predict(X).reshape(-1, 1)
        X = np.column_stack((X, km_pred))

        return(X)

    def infer(self, df):

        # preprocess inference file

        df = self.preprocessing(df)
        pred = self.model.predict(df)**2

        return(pred.copy())