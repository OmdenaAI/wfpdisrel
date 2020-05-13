import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.externals import joblib
import os
import json


class My_SVR_Model:
    def __init__(self, input_file=None):
        self.one_encoder_GENERAL_CATEGORY = OneHotEncoder()
        self.one_encoder_NATURE = OneHotEncoder()
        self.one_encoder_MONTH_END = OneHotEncoder()
        self.one_encoder_SUB_BASIN = OneHotEncoder()
        print('Current Working Directory')
        print(os.getcwd())
        # if input_file == None:
        #     self.input_file = './OUTPUT_WBI_exposer_cyclones_v14.csv'
        # else:
        #     self.input_file = input_file
        # self.init_data = pd.read_csv(input_file, sep=';')
        self.init_data = input_file.copy()
        self.num_features = ['TOTAL_HOURS_IN_LAND', 'MAX_WIND', 'MIN_PRES', 'MIN_DIST2LAND', 'V_LAND_KN',
               'RURAL_POP(%)', 'HDI', 'GDP per capita (constant 2010 US$)', 'Net flows from UN agencies US$',
               "Adjusted savings: education expenditure (% of GNI)", 'TOTAL_AFFECTED']
        self.cat_features = ['GENERAL_CATEGORY', 'MONTH_END', 'SUB BASIN', 'NATURE']

        self.one_encoder_GENERAL_CATEGORY.fit(self.init_data[['GENERAL_CATEGORY']])
        self.one_encoder_MONTH_END.fit(self.init_data[['MONTH_END']])
        self.one_encoder_NATURE.fit(self.init_data[['NATURE']])
        self.one_encoder_SUB_BASIN.fit(self.init_data[['SUB BASIN']])
        self.ss_scaler = StandardScaler()
    
    def preprocessing(self):

        data = self.init_data[self.num_features + self.cat_features]
        df_GC = pd.DataFrame(self.one_encoder_GENERAL_CATEGORY.transform(data[['GENERAL_CATEGORY']]).toarray(),
            columns=self.one_encoder_GENERAL_CATEGORY.get_feature_names())
        df_NA = pd.DataFrame(self.one_encoder_NATURE.transform(data[['NATURE']]).toarray(),
            columns=self.one_encoder_NATURE.get_feature_names())
        df_MS = pd.DataFrame(self.one_encoder_MONTH_END.transform(data[['MONTH_END']]).toarray(),
            columns=self.one_encoder_MONTH_END.get_feature_names())
        df_SB = pd.DataFrame(self.one_encoder_SUB_BASIN.transform(data[['SUB BASIN']]).toarray(),
            columns=self.one_encoder_SUB_BASIN.get_feature_names())
        df = pd.concat([df_GC, df_NA, df_MS, df_SB, data],axis=1)
        df.dropna(inplace = True)


        X = df.drop('TOTAL_AFFECTED',axis=1)
        y = df['TOTAL_AFFECTED']
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        for train_index, test_index in sss.split(df, df['GENERAL_CATEGORY']):
            train_x, test_x = X.iloc[train_index], X.iloc[test_index]
            train_y, test_y = y.iloc[train_index], y.iloc[test_index]
    
        train_x.drop(columns = self.cat_features, axis=1, inplace=True)
        test_x.drop(columns= self.cat_features, axis=1, inplace=True)

        train_x['MIN_PRES'] = train_x['MIN_PRES'].astype('float64')
        test_x['MIN_PRES'] = test_x['MIN_PRES'].astype('float64')

        train_x['MIN_DIST2LAND'] = train_x['MIN_DIST2LAND'].astype('float64')
        test_x['MIN_DIST2LAND'] = test_x['MIN_DIST2LAND'].astype('float64')

        cols_man = ['TOTAL_HOURS_IN_LAND', 'MAX_WIND', 'MIN_PRES', 'MIN_DIST2LAND', 'V_LAND_KN', 'RURAL_POP(%)', 'HDI', 'GDP per capita (constant 2010 US$)', 'Net flows from UN agencies US$', 'Adjusted savings: education expenditure (% of GNI)']
        for col in cols_man:
            train_x[col] = self.ss_scaler.fit_transform(train_x[[col]])
            test_x[col] = self.ss_scaler.transform(test_x[[col]])
            # joblib.dump(ss_scaler, f'std_scaler_{col}.bin', compress=True)
    
        self.train_model(train_x, train_y, test_x, test_y)

    def train_model(self, train_x, train_y, test_x, test_y):
        svm_param = {'kernel' : ('linear', 'poly', 'rbf'),
         'C' : [1,5,10,100],
         'degree' : [3,8],
         'coef0' : [0.01,10,0.5],
         'gamma' : ('auto','scale')},
        svr = SVR()
        grid_search_svr = GridSearchCV(svr, svm_param, n_jobs=-1)
        grid_search_svr.fit(train_x, train_y)
        y_pred_man_svr = grid_search_svr.predict(test_x)
        mae = mean_absolute_error(y_pred_man_svr, test_y)
        mse = mean_squared_error(y_pred_man_svr, test_y)
        rmse = np.sqrt(mean_squared_error(y_pred_man_svr,test_y))
        print(f'mean absolute error : {mae}')
        print(f'mean squared error : {mse}')
        print(f'root mean squared error : {rmse}')
        data = {
            "svr" : {
                "mae" : mae,
                "mse" : mse,
                "rmse" : rmse
            }
        }
        with open('svr-data.json','w') as outfile:
            json.dump(data,outfile)
        joblib.dump(grid_search_svr, 'svr_model.pkl')

    def infer(self,df):
        pickleFile = 'svr_model.pkl'
        if os.path.exists(pickleFile):
            pass
        else:
            self.preprocessing()

        model = joblib.load(pickleFile)
        num_features = self.num_features[:-1]
        data = df[num_features + self.cat_features].copy()
        df_GC = pd.DataFrame(self.one_encoder_GENERAL_CATEGORY.transform(data[['GENERAL_CATEGORY']]).toarray(),
            columns=self.one_encoder_GENERAL_CATEGORY.get_feature_names())
        df_NA = pd.DataFrame(self.one_encoder_NATURE.transform(data[['NATURE']]).toarray(),
            columns=self.one_encoder_NATURE.get_feature_names())
        df_MS = pd.DataFrame(self.one_encoder_MONTH_END.transform(data[['MONTH_END']]).toarray(),
            columns=self.one_encoder_MONTH_END.get_feature_names())
        df_SB = pd.DataFrame(self.one_encoder_SUB_BASIN.transform(data[['SUB BASIN']]).toarray(),
            columns=self.one_encoder_SUB_BASIN.get_feature_names())
        df = pd.concat([df_GC, df_NA, df_MS, df_SB, data],axis=1)
        df.drop(self.cat_features, axis=1, inplace=True)

        ### Implementation of Standard scaler and inference
        df['MIN_PRES'] = df['MIN_PRES'].astype('float64')
        df['MIN_DIST2LAND'] = df['MIN_DIST2LAND'].astype('float64')
        cols_man = ['TOTAL_HOURS_IN_LAND', 'MAX_WIND', 'MIN_PRES', 'MIN_DIST2LAND', 'V_LAND_KN', 'RURAL_POP(%)', 'HDI', 'GDP per capita (constant 2010 US$)', 'Net flows from UN agencies US$', 'Adjusted savings: education expenditure (% of GNI)']
        for col in cols_man:
            self.ss_scaler.fit(self.init_data[[col]])
            df[col] = self.ss_scaler.transform(df[[col]])
        print(df.info())
        y_pred = model.predict(df)
        return y_pred.copy()

# def preprocessing(inputFile = None):
#     one_encoder_GENERAL_CATEGORY = OneHotEncoder()
#     one_encoder_NATURE = OneHotEncoder()
#     one_encoder_MONTH_END = OneHotEncoder()
#     one_encoder_SUB_BASIN = OneHotEncoder()
#     if inputFile == None:
#         inputFile = '../OUTPUT_WBI_exposer_cyclones_v14.csv'
#     init_data = pd.read_csv(inputFile)
#     num_features = ['TOTAL_HOURS_IN_LAND', 'MAX_WIND', 'MIN_PRES', 'MIN_DIST2LAND', 'V_LAND_KN',
#                'RURAL_POP(%)', 'HDI', 'GDP per capita (constant 2010 US$)', 'Net flows from UN agencies US$',
#                "Adjusted savings: education expenditure (% of GNI)", 'TOTAL_AFFECTED']
#     cat_features = ['GENERAL_CATEGORY', 'MONTH_END', 'SUB BASIN', 'NATURE']
#     data = init_data[num_features + cat_features]
#     df_GC = pd.DataFrame(one_encoder_GENERAL_CATEGORY.fit_transform(data[['GENERAL_CATEGORY']]).toarray(),
#     columns=one_encoder_GENERAL_CATEGORY.get_feature_names())
#     df_NA = pd.DataFrame(one_encoder_NATURE.fit_transform(data[['NATURE']]).toarray(),
#     columns=one_encoder_NATURE.get_feature_names())
#     df_MS = pd.DataFrame(one_encoder_MONTH_END.fit_transform(data[['MONTH_END']]).toarray(),
#     columns=one_encoder_MONTH_END.get_feature_names())
#     df_SB = pd.DataFrame(one_encoder_SUB_BASIN.fit_transform(data[['SUB BASIN']]).toarray(),
#     columns=one_encoder_SUB_BASIN.get_feature_names())
#     df = pd.concat([df_GC, df_NA, df_MS, df_SB, data],axis=1)
#     df.dropna(inplace = True)

#     X = df.drop('TOTAL_AFFECTED',axis=1)
#     y = df['TOTAL_AFFECTED']
#     sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
#     for train_index, test_index in sss.split(df, df['GENERAL_CATEGORY']):
#         train_x, test_x = X.iloc[train_index], X.iloc[test_index]
#         train_y, test_y = y.iloc[train_index], y.iloc[test_index]
    
#     train_x.drop(columns = cat_features, axis=1, inplace=True)
#     test_x.drop(columns= cat_features, axis=1, inplace=True)

#     ss_scaler = StandardScaler()
#     train_x['MIN_PRES'] = train_x['MIN_PRES'].astype('float64')
#     test_x['MIN_PRES'] = test_x['MIN_PRES'].astype('float64')

#     train_x['MIN_DIST2LAND'] = train_x['MIN_DIST2LAND'].astype('float64')
#     test_x['MIN_DIST2LAND'] = test_x['MIN_DIST2LAND'].astype('float64')

#     cols_man = train_x.select_dtypes(include=['float64']).columns.to_list()
#     for col in cols_man:
#         train_x[col] = ss_scaler.fit_transform(train_x[[col]])
#         test_x[col] = ss_scaler.transform(test_x[[col]])
#         # joblib.dump(ss_scaler, f'std_scaler_{col}.bin', compress=True)
    
#     train_model(train_x, train_y, test_x, test_y)


# def train_model(train_x, train_y, test_x, test_y):
#     svm_param = {'kernel' : ('linear', 'poly', 'rbf'),
#          'C' : [1,5,10,100],
#          'degree' : [3,8],
#          'coef0' : [0.01,10,0.5],
#          'gamma' : ('auto','scale')},
#     svr = SVR()
#     grid_search_svr = GridSearchCV(svr, svm_param, n_jobs=-1)
#     grid_search_svr.fit(train_x, train_y)
#     y_pred_man_svr = grid_search_svr.predict(test_x)
#     print(f'mean absolute error : {mean_absolute_error(y_pred_man_svr, test_y)}')
#     print(f'mean squared error : {mean_squared_error(y_pred_man_svr, test_y)}')
#     print(f'root mean squared error : {np.sqrt(mean_squared_error(y_pred_man_svr,test_y))}')
#     joblib.dump(grid_search_svr, 'model.pkl')


# def infer(df):
    # pickleFile = 'model.pkl'
    # if os.path.exists(pickleFile):
    #     pass
    # else:
    #     preprocessing()
    # model = joblib.load(pickleFile)

    # # Preprocessing for inference
    # one_encoder_GENERAL_CATEGORY = OneHotEncoder()
    # one_encoder_NATURE = OneHotEncoder()
    # one_encoder_MONTH_END = OneHotEncoder()
    # one_encoder_SUB_BASIN = OneHotEncoder()
    # inputFile = '../OUTPUT_WBI_exposer_cyclones_v14.csv'
    # init_data = pd.read_csv(inputFile)
    # one_encoder_GENERAL_CATEGORY.fit(init_data[['GENERAL_CATEGORY']])
    # one_encoder_MONTH_END.fit(init_data[['MONTH_END']])
    # one_encoder_NATURE.fit(init_data[['NATURE']])
    # one_encoder_SUB_BASIN.fit(init_data[['SUB BASIN']])

    # num_features = ['TOTAL_HOURS_IN_LAND', 'MAX_WIND', 'MIN_PRES', 'MIN_DIST2LAND', 'V_LAND_KN',
    #            'RURAL_POP(%)', 'HDI', 'GDP per capita (constant 2010 US$)', 'Net flows from UN agencies US$',
    #            "Adjusted savings: education expenditure (% of GNI)", 'TOTAL_AFFECTED']
    # cat_features = ['GENERAL_CATEGORY', 'MONTH_END', 'SUB BASIN', 'NATURE']
    # data = df[num_features + cat_features]
    # df_GC = pd.DataFrame(one_encoder_GENERAL_CATEGORY.fit_transform(data[['GENERAL_CATEGORY']]).toarray(),
    # columns=one_encoder_GENERAL_CATEGORY.get_feature_names())
    # df_NA = pd.DataFrame(one_encoder_NATURE.fit_transform(data[['NATURE']]).toarray(),
    # columns=one_encoder_NATURE.get_feature_names())
    # df_MS = pd.DataFrame(one_encoder_MONTH_END.fit_transform(data[['MONTH_END']]).toarray(),
    # columns=one_encoder_MONTH_END.get_feature_names())
    # df_SB = pd.DataFrame(one_encoder_SUB_BASIN.fit_transform(data[['SUB BASIN']]).toarray(),
    # columns=one_encoder_SUB_BASIN.get_feature_names())
    # df = pd.concat([df_GC, df_NA, df_MS, df_SB, data],axis=1)

    # ### Implementation of Standard scaler and inference
    # ss_scaler = StandardScaler()
    # df['MIN_PRES'] = df['MIN_PRES'].astype('float64')
    # df['MIN_DIST2LAND'] = df['MIN_DIST2LAND'].astype('float64')
    # cols_man = df.select_dtypes(include=['float64']).columns.to_list()
    # for col in cols_man:
    #     ss_scaler.fit(init_data[[col]])
    #     df[col] = ss_scaler.transform(df[[col]])

    # y_pred = model.predict(df)
    # return y_pred.copy()