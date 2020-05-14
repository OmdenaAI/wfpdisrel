

import numpy as np


import reverse_geocoder as rg


median_cols = ['MAX_STORMSPEED', 
'Arable land (hectares per person)', 
'Food production index (2004-2006 = 100)', 
'GDP per capita (constant 2010 US$)', 
'Life expectancy at birth, total (years)',
'Adjusted savings: education expenditure (% of GNI)',
'Cereal yield (kg per hectare)',
'MIN_PRES',
'RURAL_POP',
'Arable land (hectares per person)', 
'POP_MAX_34_ADJ', 
'POP_MAX_50_ADJ', 
'POP_MAX_64_ADJ', 
]

dummy_cols = ['MONTH_START_1', 'MONTH_START_2', 'MONTH_START_3', 'MONTH_START_4',
       'MONTH_START_5', 'MONTH_START_6', 'MONTH_START_7', 'MONTH_START_8',
       'MONTH_START_9', 'MONTH_START_10', 'MONTH_START_11',
       'MONTH_START_12', 'MAX_USA_SSHS_INLAND_-7',
       'MAX_USA_SSHS_INLAND_-6', 'MAX_USA_SSHS_INLAND_-5',
       'MAX_USA_SSHS_INLAND_-4', 'MAX_USA_SSHS_INLAND_-3',
       'MAX_USA_SSHS_INLAND_-2', 'MAX_USA_SSHS_INLAND_-1',
       'MAX_USA_SSHS_INLAND_0', 'MAX_USA_SSHS_INLAND_1',
       'MAX_USA_SSHS_INLAND_2', 'MAX_USA_SSHS_INLAND_3',
       'MAX_USA_SSHS_INLAND_4', 'MAX_USA_SSHS_INLAND_5', 'NEW_GEN_CAT_0',
       'NEW_GEN_CAT_2', 'NEW_BASIN_0', 'NEW_BASIN_2', 'NEW_SUB BASIN_0',
       'NEW_SUB BASIN_3', 'NEW_Income_level_Final_0',
       'NEW_Income_level_Final_2', 'SUBGEN_0', 'SUBGEN_2', 'SUBGEN_3',
       'SUBGEN_5', 'SUBINCOME_0', 'SUBINCOME_2', 'SUBINCOME_3',
       'SUBINCOME_5']

object_cols = ['MONTH_START',
 'MAX_USA_SSHS_INLAND',
 'NEW_GEN_CAT',
 'NEW_BASIN',
 'NEW_SUB BASIN',
 'NEW_Income_level_Final',
 'SUBGEN',
 'SUBINCOME']

features = ['season',
 'TOTAL_HOURS_IN_LAND',
 'MAX_WIND',
 'MIN_PRES',
 'MIN_DIST2LAND',
 'MAX_STORMSPEED',
 'MAX_USA_SSHS',
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
 'pop_break',
 'pop_break_two',
 'Expectancy_break',
 'cereal_break',
 'cereal_break_two',
 'rural_break',
 'rural_break_two',
 'rural_break_three',
 'rural_break_four',
 'dis2land_bin',
 'SSH_Freq',
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
 'MAX_USA_SSHS_INLAND_-7',
 'MAX_USA_SSHS_INLAND_-6',
 'MAX_USA_SSHS_INLAND_-5',
 'MAX_USA_SSHS_INLAND_-4',
 'MAX_USA_SSHS_INLAND_-3',
 'MAX_USA_SSHS_INLAND_-2',
 'MAX_USA_SSHS_INLAND_-1',
 'MAX_USA_SSHS_INLAND_0',
 'MAX_USA_SSHS_INLAND_1',
 'MAX_USA_SSHS_INLAND_2',
 'MAX_USA_SSHS_INLAND_3',
 'MAX_USA_SSHS_INLAND_4',
 'MAX_USA_SSHS_INLAND_5',
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

log_cols = ['TOTAL_HOURS_IN_LAND', 
            'POP_DEN_SQ_KM', 'Arable land (hectares per person)',
            'POP_MAX_34_ADJ', 'POP_MAX_50_ADJ', 'POP_MAX_64_ADJ']

def new_cats(df):
### Final Trans

    ## USA_SSHS Preprocessing

    df['MAX_USA_SSHS_INLAND'][df['MAX_USA_SSHS_INLAND'] == 'No landing'] = -7
    df['MAX_USA_SSHS_INLAND'] = df['MAX_USA_SSHS_INLAND'].astype('int')

    ## Engineer new features
    # If value is 7
    # And if value falls in the SS scale

    df['MAX_SSH_7'] = np.where(df['MAX_USA_SSHS_INLAND'] == -7, 1, 0)
    df['MAX_SSH_SS'] = np.where(df['MAX_USA_SSHS_INLAND'] > 0, 1, 0)


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



# Model Architecture
def build_model():
  clf = Sequential()

  clf.add(Dense(32, kernel_initializer='random_normal', input_shape=(scaled_X_Kmeans.shape[1], )))
  clf.add(BatchNormalization())
  clf.add(ReLU())

  clf.add(Dense(64, kernel_initializer='random_normal'))
  clf.add(ReLU())
  clf.add(Dropout(0.45))
  
  clf.add(Dense(128, kernel_initializer='random_normal'))
  clf.add(BatchNormalization())
  clf.add(ReLU())

  clf.add(Dense(256, kernel_initializer='random_normal'))
  clf.add(BatchNormalization())
  clf.add(ReLU())

  clf.add(Dense(512, kernel_initializer='random_normal'))
  clf.add(BatchNormalization())
  clf.add(ReLU())

  clf.add(Dense(512, kernel_initializer='random_normal'))
  clf.add(BatchNormalization()) ## 512 ----
  clf.add(ReLU())

  clf.add(Dense(1, kernel_initializer='random_normal'))

  optim = optimizers.RMSprop(learning_rate=0.001)
  clf.compile(loss='mse', optimizer=optim, metrics=['mae'])

  return(clf)