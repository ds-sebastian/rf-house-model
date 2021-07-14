#%%
import evalml
from evalml.automl import AutoMLSearch
from evalml.preprocessing import split_data
import woodwork as ww
import pandas as pd
import numpy as np

df = pd.read_csv(r'.\data\Model_Data.csv', low_memory=False)
df = df.replace(',','', regex=True)
df = df.convert_dtypes().replace({pd.NA: np.nan})

remove_home_types = ['Vacant Land', 'Mobile/Manufactured Home']
df = df.loc[~df["PROPERTY TYPE"].isin(remove_home_types)]
df['DATE SOLD_UNIX'] = (pd.to_datetime(df['SOLD DATE']) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s') #76
df = df[pd.to_datetime(df['SOLD DATE']).dt.year >= 2020]
df.drop(['SOLD DATE', 'STATE OR PROVINCE', 'Unnamed: 0', 'ZIP OR POSTAL CODE','CITY', 'YEAR BUILT', 'url'], axis=1, inplace=True)
df['LATITUDE'] = df['LATITUDE'].astype('float64', copy=False)
df['LONGITUDE'] = df['LONGITUDE'].astype('float64', copy=False)
df = df[df['PRICE'] < 1000000]
#%%
X = df.drop(columns=['PRICE'])
X.ww.init()

# This tells you the number of missing values 
na_count = X.ww.describe().loc['nan_count']
columns_with_missing_values = na_count[na_count > 0].index

# This tells you the columns that are natural language
nat_lang_columns = X.ww.select("NaturalLanguage").columns

nat_lang_with_nan = set(nat_lang_columns).intersection(columns_with_missing_values)
y= np.log(df['PRICE']) 
X.ww.set_types({col: "Categorical" for col in nat_lang_with_nan})

X_train, X_holdout, y_train, y_holdout = split_data(X, y, problem_type='regression', test_size=.2)

#%%
automl = AutoMLSearch(X_train=X_train,
                    y_train=y_train, 
                    problem_type='regression',
                    ensembling=True,
                    max_time = '420 min',
                    objective = 'root mean squared error',
                    allowed_model_families = ['LIGHTGBM','RANDOM_FOREST','XGBOOST','CATBOOST'],
                    patience = 200)
automl.search()
# %%

pipeline = automl.best_pipeline

#%%
automl.describe_pipeline(pipeline)

# X is your feature matrix

# %%

print(pipeline.parameters)
# %%
#automl.scores(X_holdout, y_holdout)
pipeline.parameters
# %%
pipeline.graph()
# %%
pipeline.score(X_holdout, y_holdout, objectives = ["root mean squared error"])
# %%
automl.rankings
# %%
#{'Imputer': {'categorical_impute_strategy': 'most_frequent',
#  'numeric_impute_strategy': 'mean',
#  'categorical_fill_value': None,
#  'numeric_fill_value': None},
# 'One Hot Encoder': {'top_n': 10,
#  'features_to_encode': None,
#  'categories': None,
#  'drop': 'if_binary',
#  'handle_unknown': 'ignore',
#  'handle_missing': 'error'},
# 'XGBoost Regressor': {'eta': 0.03842638804730825,
#  'max_depth': 12,
#  'min_child_weight': 9.895364542533038,
#  'n_estimators': 965,
#  'n_jobs': -1}}