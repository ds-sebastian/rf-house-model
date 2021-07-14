#%%
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

import datetime

import regex as re

#%%
def sales_clean(sales_data):

    sales_data['url']=sales_data['URL (SEE http://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING)']
    sales_data.drop(['SALE TYPE', 'ADDRESS','LOCATION','$/SQUARE FEET', 'STATUS', 'URL (SEE http://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING)',
                  'NEXT OPEN HOUSE START TIME', 'NEXT OPEN HOUSE END TIME',
                  'SOURCE','MLS#','FAVORITE', 'INTERESTED'],axis=1,inplace=True)


    sales_data = sales_data[sales_data['PRICE'].notna()]
    sales_data = sales_data.convert_dtypes()

    sales_data.dtypes


    sales_data['SOLD DATE']= pd.to_datetime(sales_data['SOLD DATE'], errors='coerce')

    sales_data['Age'] = sales_data['SOLD DATE'].apply(lambda x: x.year)-sales_data['YEAR BUILT']
    sales_data['Age'] = sales_data['Age'].replace({pd.NA: np.nan})

    #City name unification
    sales_data['CITY'] = sales_data['CITY'].str.upper()

    #Zip
    sales_data['ZIP OR POSTAL CODE'] = sales_data['ZIP OR POSTAL CODE'].str[:5]

    #PROPERTY TYPE
    sales_data_types = ['Unknown','Other', 'Ranch', 'Parking']
    sales_data.loc[sales_data["PROPERTY TYPE"].isin(sales_data_types), "PROPERTY TYPE"] = "Other"

    sales_data_types = ['Multi-Family (2-4 Unit)','Multi-Family (5+ Unit)']
    sales_data.loc[sales_data["PROPERTY TYPE"].isin(sales_data_types), "PROPERTY TYPE"] = "Multi-Family"

    #fill Lot Size of condo and mobile/manufactured home to 0
    sales_data.loc[(sales_data['LOT SIZE'].isna()) & (sales_data['PROPERTY TYPE']=='Mobile/Manufactured Home')]['PROPERTY TYPE'] = 0
    sales_data.loc[(sales_data['LOT SIZE'].isna()) & (sales_data['PROPERTY TYPE']=='Condo/Co-op')]['LOT SIZE'] = 0
    sales_data.loc[(sales_data['LOT SIZE'].isna()) & (sales_data['PROPERTY TYPE']=='Townhouse')]['LOT SIZE'] = 0
    sales_data['PROPERTY TYPE'].value_counts()

    sales_data = sales_data.convert_dtypes()

    return sales_data


def mls_clean(mls_data):
    kept_cols = ['url',
                'numStories',
                'yearRenovated',
                'sqFtFinished',
                'totalSqFt',
                'lotSqFt',
                #'propertyLastUpdatedDate', #Could use to clean data?
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
                'taxesDue']
    initial_df = pd.DataFrame(columns=kept_cols)
    mls_data = pd.concat([initial_df,mls_data])[kept_cols]

    return mls_data

#%%