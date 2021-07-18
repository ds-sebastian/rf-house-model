#%%
## Libraries:

import pandas as pd  
import numpy as np
import seaborn as sns
import sklearn.metrics
from sklearn.model_selection import RandomizedSearchCV, train_test_split
#from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from xgboost import plot_importance
import matplotlib.pyplot as plt
#from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform
from sklearn.feature_selection._base import SelectorMixin
from sklearn.feature_extraction.text import _VectorizerMixin

from functions import get_feature_names
from data.functions.selenium_funcs import mls_parse
#Tune hyper paramters?
hyper_param_tune = False
#%%%
df = pd.read_csv(r'.\data\Model_Data.csv')

#%%
categorical_features = ['PROPERTY TYPE',
                #, 'CITY'
                #, 'kmn_cluster'
                #, 'ZIP OR POSTAL CODE'
                'AIR_CONDITIONING_CODE', 
                'FUEL_CODE', 
                'HEATING_TYPE_CODE',
                'EXTERIOR_WALL_CODE', 
                'ROOF_TYPE_CODE', 
                'GARAGE_CODE',
                #'MUNICIPALITY_NAME', 
                #'ZONING_CODE', 
                #'COUNTY_USE_DESCRIPTION', 
                ]


#%%
#df = df[pd.to_datetime(df['SOLD DATE']).dt.year >= 2020]
df.drop(['Unnamed: 0'], axis=1, inplace=True)
df = df[df['PRICE'] < 1000000]
#%%

print(df.nunique())
print(df.info())

#del test,start
#%%

X = df.drop(columns=['PRICE'])
y= np.log(df['PRICE']) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


#%%

categorical_transformer = Pipeline(steps=[
    ('imputer_cat', SimpleImputer(strategy='most_frequent'#, fill_value='Unknown'
                                                    )),
    ('ohe', OneHotEncoder(handle_unknown = 'ignore'))])


numeric_features = X.drop(columns=categorical_features).columns.values
numeric_transformer = Pipeline(steps=[
    ('imputer_num', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])



preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

#%%

def get_feature_out(estimator, feature_in):
    if hasattr(estimator, 'get_feature_names'):
        if isinstance(estimator, _VectorizerMixin):
            # handling all vectorizers
            return [f'vec_{f}'
                    for f in estimator.get_feature_names()]
        else:
            return estimator.get_feature_names(feature_in)
    elif isinstance(estimator, SelectorMixin):
        return np.array(feature_in)[estimator.get_support()]
    else:
        return feature_in


def get_ct_feature_names(ct):
    # handles all estimators, pipelines inside ColumnTransfomer
    # doesn't work when remainder =='passthrough'
    # which requires the input column names.
    output_features = []

    for name, estimator, features in ct.transformers_:
        if name != 'remainder':
            if isinstance(estimator, Pipeline):
                current_features = features
                for step in estimator:
                    current_features = get_feature_out(step, current_features)
                features_out = current_features
            else:
                features_out = get_feature_out(estimator, features)
            output_features.extend(features_out)
        elif estimator == 'passthrough':
            output_features.extend(ct._feature_names_in[features])

    return output_features
#%%

X_train = preprocessor.fit_transform(X_train)

try:
    X_train = pd.DataFrame(X_train,
                             columns=get_ct_feature_names(preprocessor))
except:
    X_train = pd.DataFrame.sparse.from_spmatrix(X_train,
                             columns=get_ct_feature_names(preprocessor))


#%%
X_test = preprocessor.transform(X_test)

try:
    X_test = pd.DataFrame(X_test, columns=get_ct_feature_names(preprocessor))
except: 
    X_test = pd.DataFrame.sparse.from_spmatrix(X_test, columns=get_ct_feature_names(preprocessor))



#%%
import json
with open('regression_best_params.json') as json_file:
    reg_best_params = json.load(json_file)

#%%
        


model = XGBRegressor(**reg_best_params
                       ,tree_method='gpu_hist', gpu_id = 0
                       ,random_state = 42
                    )

#Tuning
if hyper_param_tune == True:
    param_dist = {'n_estimators': randint(750, 1150),
                  'learning_rate': uniform(0.02, 0.04),
                  'subsample': uniform(0.4, 0.6),
                  'max_depth': [10, 12, 14],
                  'colsample_bytree': uniform(0.01, 0.99),
                  'min_child_weight': uniform(7, 11)
                 }  
    model = RandomizedSearchCV(model
                                , param_dist
                                , random_state=42
                                )


#Fit model
eval_set = [(X_train, y_train), (X_test, y_test)]

model.fit(X_train, y_train
        , eval_metric="rmse", eval_set=eval_set
        , early_stopping_rounds=300
        )

if hyper_param_tune == True:
    jsn = json.dumps(model.best_params_)
    f = open("regression_best_params.json","w")
    f.write(jsn)
    f.close()


#%%
y_pred = model.predict(X_test)

mse=sklearn.metrics.mean_squared_error(np.exp(y_test), np.exp(y_pred))
rmse = mse**(1/2.0)
print(rmse)


#%% 
if hyper_param_tune == False:

    ax = plot_importance(model)
    fig = ax.figure
    fig.set_size_inches(10, 25)
    fig.tight_layout()
    fig.savefig('xgboost feature importance.png')

# %%
from sklearn.feature_selection import SelectFromModel
feat_compare = pd.DataFrame()

#60 features is a good cutoff
thresholds = np.array([10,11,12,13,14,15])
for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(model, max_features=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    select_X_test = selection.transform(X_test)
    select_eval_set = [(select_X_train, y_train), (select_X_test, y_test)]
    # train model
    selection_model = XGBRegressor(**reg_best_params
                       ,tree_method='gpu_hist', gpu_id = 0
                       ,random_state = 42
                    )
    selection_model.fit(select_X_train, y_train
        , eval_metric="rmse", eval_set=select_eval_set
        , early_stopping_rounds=300)
    # eval model
    select_X_test = selection.transform(X_test)
    predictions = selection_model.predict(select_X_test)
    rmse = sklearn.metrics.mean_squared_error(np.exp(y_test), np.exp(predictions))**(1/2.000)
    print("Thresh=%.3f, n=%d, RMSE: %d" % (thresh, select_X_train.shape[1], rmse))
    feat_compare = feat_compare.append({'thresh': thresh, 'rmse': rmse}, ignore_index = True) 

plt.figure(figsize=(10,10))
plt.scatter(feat_compare['thresh'].values, feat_compare['rmse'].values, c='crimson')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.show()


#%%

selection = SelectFromModel(model, max_features=15, prefit=True)
select_X_train = selection.transform(X_train)
select_X_test = selection.transform(X_test)


select_eval_set = [(select_X_train, y_train), (select_X_test, y_test)]

# train model
selection_model = XGBRegressor(**reg_best_params
                   ,tree_method='gpu_hist', gpu_id = 0
                   ,random_state = 42
                )
selection_model.fit(select_X_train, y_train
        , eval_metric="rmse", eval_set=select_eval_set
        , early_stopping_rounds=300)
# eval model

predictions = selection_model.predict(select_X_test)
rmse = sklearn.metrics.mean_squared_error(np.exp(y_test), np.exp(predictions))**(1/2.000)
print("Thresh=%.3f, n=%d, RMSE: %d" % (thresh, select_X_train.shape[1], rmse))


#%%
user_agent_list = open(r"G:\DataScience\MortgageCalc\house-model\user-agents.txt", "r").read().splitlines()

redfin_data = mls_parse(r'https://www.redfin.com/NC/Holly-Springs/321-Stone-Hedge-Ct-27540/home/41232040', user_agent_list)

redfin_data = redfin_data[[
#'url',
'numStories',
'yearRenovated',
'sqFtFinished',
'totalSqFt',
'lotSqFt',
#'propertyLastUpdatedDate',
'FULL_BATHS',
'AIR_CONDITIONING_CODE', #
'FUEL_CODE', #
'HEATING_TYPE_CODE',#
'EXTERIOR_WALL_CODE', #
'ROOF_TYPE_CODE', #
'ASSESSED_YEAR',
#'SUBDIVISION_NAME', #
'LIVING_SQUARE_FEET',
'BUILDING_SQUARE_FEET',
'GARAGE_PARKING_SQUARE_FEET',
'GARAGE_CODE',
'NUMBER_OF_BUILDINGS',
'LAND_SQUARE_FOOTAGE',
'MUNICIPALITY_NAME', #
'ACRES',
'ZONING_CODE', #
'COUNTY_USE_DESCRIPTION', #
'parentRating',
'schooldisance',
'servesHome',
'schoolchoice',
'numberOfStudents',
'school_score',
'student_teacher_ratio',
'review_nums',
'taxableLandValue',
'taxableImprovementValue',
'rollYear',
'taxesDue']]
#%%

predict = pd.DataFrame({'Unnamed: 0': 9999 ,
'SOLD DATE': '2021-07-01' ,
'PROPERTY TYPE': 'Single Family Residential' ,
'CITY': 'HOLLY SPRINGS' ,
'STATE OR PROVINCE': 'NC' ,
'ZIP OR POSTAL CODE': 27540 ,
'PRICE': 0 ,
'BEDS': 3 ,
'BATHS': 2.5 ,
'SQUARE FEET': 1772 ,
'LOT SIZE': 6098 ,
'YEAR BUILT': 1998 ,
'DAYS ON MARKET': 1 ,
'HOA/MONTH': 33,
'LATITUDE': 35.656310 ,
'LONGITUDE': -78.840714 ,
'Sold Year': 2021 ,
'Age': 23
}, index=[0]).convert_dtypes()


#%%

predict = pd.concat([predict,redfin_data], axis =1)
predict = predict.replace(',','', regex=True).convert_dtypes()

#%%
predict['DATE SOLD_UNIX'] = (pd.to_datetime(predict['SOLD DATE']) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s') #76
#predict['kmn_cluster'] = kmeans.predict(predict[['LATITUDE','LONGITUDE']])
#predict['is_Pandemic'] = True
#predict['is_CY'] = True
#%%
predict.drop(['Sold Year','SOLD DATE', 'STATE OR PROVINCE', 'Unnamed: 0','ZIP OR POSTAL CODE', 'CITY', 'YEAR BUILT', 'MUNICIPALITY_NAME', 'ZONING_CODE' ], axis=1, inplace=True)
#predict.drop(['CITY'], axis=1, inplace=True)
predict.drop(columns=['PRICE'], inplace=True)
#predict['ZIP OR POSTAL CODE'] = predict['ZIP OR POSTAL CODE'].astype(str).str[:3].astype(int)

#%%

try:
    predict = pd.DataFrame.sparse.from_spmatrix(preprocessor.transform(predict), columns = get_feature_names(preprocessor))
except:
    predict = pd.DataFrame(preprocessor.transform(predict), columns = get_feature_names(preprocessor))

final_prediction = np.exp(model.predict(predict))

print('Predicted Sales Price:', final_prediction)
# %%

# %%

