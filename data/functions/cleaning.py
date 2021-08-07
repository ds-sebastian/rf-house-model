# %%
import pandas as pd
import numpy as np

#%%
def sales_clean(sales_data):
    '''cleans sales data'''
    sales_data = sales_data[sales_data['PRICE'].notna()]
    sales_data = sales_data.replace(',','', regex=True)
    sales_data = sales_data.replace({pd.NA: np.nan})
    sales_data = sales_data.applymap(lambda s:s.upper() if type(s) == str else s)
    sales_data.columns= sales_data.columns.str.strip().str.upper()
    sales_data.rename(columns={'URL (SEE HTTP://WWW.REDFIN.COM/BUY-A-HOME/COMPARATIVE-MARKET-ANALYSIS FOR INFO ON PRICING)':'URL'}, inplace=True) #Rename url
    sales_data.drop(['SALE TYPE', 'ADDRESS', 'LOCATION', '$/SQUARE FEET', 'STATUS',
                     'NEXT OPEN HOUSE START TIME', 'NEXT OPEN HOUSE END TIME',
                     'SOURCE', 'MLS#', 'FAVORITE', 'INTERESTED'], axis=1, inplace=True)

    sales_data = sales_data[sales_data['PRICE'].notna()]

    sales_data['SOLD DATE'] = pd.to_datetime(sales_data['SOLD DATE'], errors='coerce')
    sales_data['SOLD_YEAR'] = sales_data['SOLD DATE'].dt.year
    sales_data['SOLD_MONTH'] = sales_data['SOLD DATE'].dt.month
    sales_data['SOLD_DAY'] = sales_data['SOLD DATE'].dt.day
    sales_data['DATE SOLD_UNIX'] = (sales_data['SOLD DATE'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s') #76

    sales_data['AGE'] = sales_data['SOLD DATE'].apply(
        lambda x: x.year)-sales_data['YEAR BUILT']
    # Zip simplify
    sales_data['ZIP OR POSTAL CODE'] = sales_data['ZIP OR POSTAL CODE'].str[:5]
    # Remove other property types
    sales_data_types = ['UNKNOWN', 'OTHER', 'RANCH', 'PARKING', 'VACANT LAND', 'MOBILE/MANUFACTURED HOME']
    sales_data = sales_data.loc[~sales_data["PROPERTY TYPE"].isin(sales_data_types)]
    #Multi family combine
    sales_data.loc[sales_data['PROPERTY TYPE'].str.contains('MULTI'), 'PROPERTY TYPE'] = 'MULTI-FAMILY'
    # fill Lot Size of condo and mobile/manufactured home to 0
    sales_data.loc[(sales_data['LOT SIZE'].isna()) & (sales_data['PROPERTY TYPE']=='CONDO/CO-OP'),'LOT SIZE'] = 0
    sales_data.loc[(sales_data['LOT SIZE'].isna()) & (sales_data['PROPERTY TYPE']=='TOWNHOUSE'),'LOT SIZE'] = 0
    sales_data = sales_data.convert_dtypes()
    return sales_data

#%%
def mls_clean(mls_data):
    '''cleans mls data'''
    mls_data = mls_data.replace(',','', regex=True)
    mls_data = mls_data.replace({pd.NA: np.nan})
    mls_data = mls_data.applymap(lambda s:s.upper() if type(s) == str else s)
    mls_data.columns= mls_data.columns.str.strip().str.upper()
    kept_cols = ['URL',
                 'NUMSTORIES',
                 'YEARRENOVATED',
                 'SQFTFINISHED',
                 'TOTALSQFT',
                 'LOTSQFT',
                 'FULL_BATHS',
                 'HALF_BATHS',
                 'AIR_CONDITIONING_CODE',
                 'FUEL_CODE',
                 'HEATING_TYPE_CODE',
                 'EXTERIOR_WALL_CODE',
                 'ROOF_TYPE_CODE',
                 'ASSESSED_YEAR',
                 'LIVING_SQUARE_FEET',
                 'BUILDING_SQUARE_FEET',
                 'GARAGE_PARKING_SQUARE_FEET',
                 'PARKING_TYPE',
                 'NUMBER_OF_BUILDINGS',
                 'PARENTRATING',
                 'SCHOOLDISANCE',
                 'SERVESHOME',
                 'NUMBEROFSTUDENTS',
                 'SCHOOL_SCORE',
                 'STUDENT_TEACHER_RATIO',
                 'REVIEW_NUMS',
                 'TAXABLELANDVALUE',
                 'TAXABLEIMPROVEMENTVALUE',
                 'ROLLYEAR',
                 'TAXESDUE',
                 'NUMBER_OF_FIREPLACES', 
                 'STYLE_CODE',
                 'BASEMENT_TYPE_CODE',
                 'BUILDING_CODE'
                 ]
    initial_df = pd.DataFrame(columns=kept_cols)
    mls_data = pd.concat([initial_df, mls_data])[kept_cols]
    #%%
    #Single Family?
    mls_data.loc[~mls_data['BUILDING_CODE'].isin(['SINGLE FAMILY']) & (~mls_data['BUILDING_CODE'].isna()),'BUILDING_CODE'] = 'OTHER'
    #%%
    # AIR_CONDITIONING_CODE not in Separate System,Package, Central, Heat Pump == None 
    mls_data.loc[~mls_data['AIR_CONDITIONING_CODE'].isin(['SEPARATE SYSTEM', 'PACKAGE', 'CENTRAL', 'HEAT PUMP']) & (~mls_data['AIR_CONDITIONING_CODE'].isna()),'AIR_CONDITIONING_CODE'] = 'OTHER'
    # FUEL_CODE not in Gas, Electric == Other
    mls_data.loc[~mls_data['FUEL_CODE'].isin(['GAS', 'ELECTRIC']) & (~mls_data['FUEL_CODE'].isna()),'FUEL_CODE'] = 'OTHER'
    #HEATING TYPE MAP
    mls_data['HEATING_TYPE_CODE'] = mls_data['HEATING_TYPE_CODE'].map(
            {'FORCED AIR' : 'WATER-AIR',
            'PACKAGE' : 'GAS-ELECTRIC',
            'HEAT PUMP' : 'WATER-AIR',
            'CENTRAL' : 'WATER-AIR',
            'HOT AIR' : 'WATER-AIR',
            'BASEBOARD ELECTRIC' : 'GAS-ELECTRIC',
            'FLOOR/WALL FURNACE' : 'GAS-ELECTRIC',
            'PARTIAL' : 'NONE',
            'SPACE' : 'NONE',
            'NOT DUCTED' : 'NONE',
            'HEAT PUMP ELECTRIC' : 'GAS-ELECTRIC',
            'BASEBOARD HOT WATER' : 'WATER-AIR',
            'RADIANT' : 'GAS-ELECTRIC',
            'STEAM HOT WATER' : 'WATER-AIR',
            'WALL' : 'GAS-ELECTRIC',
            'HOT WATER' : 'WATER-AIR',
            'UNIT' : 'GAS-ELECTRIC',
            'FORCED AIR NOT DUCTED' : 'WATER-AIR',
            'WALL ELECTRIC' : 'GAS-ELECTRIC',
            'GAS' : 'GAS-ELECTRIC',
            'ELECTRIC' : 'GAS-ELECTRIC',
            'BASEBOARD' : 'GAS-ELECTRIC',
            'RADIANT CEILING' : 'GAS-ELECTRIC',
            'SOLAR' : 'GAS-ELECTRIC',
            'FORCED AIR GAS' : 'GAS-ELECTRIC',
            'RADIANT HOT WATER' : 'WATER-AIR',
            'WALL FURNACE' : 'NONE',
            'WOOD STOVE' : 'NONE',
            'STEAM' : 'WATER-AIR',
            'RADIANT ELECTRIC' : 'GAS-ELECTRIC'
            }, na_action='ignore')
    #Keep top 10 wall types
    mls_data.loc[~mls_data['EXTERIOR_WALL_CODE'].isin(list(mls_data['EXTERIOR_WALL_CODE'].value_counts().head(10).index)) & (~mls_data['EXTERIOR_WALL_CODE'].isna()),'EXTERIOR_WALL_CODE'] = 'OTHER'
    #keep top 3 roof types
    mls_data.loc[~mls_data['ROOF_TYPE_CODE'].isin(list(mls_data['ROOF_TYPE_CODE'].value_counts().head(3).index)) & (~mls_data['ROOF_TYPE_CODE'].isna()),'ROOF_TYPE_CODE'] = 'OTHER'
    #keep top 7 Style codes
    mls_data.loc[~mls_data['STYLE_CODE'].isin(list(mls_data['STYLE_CODE'].value_counts().head(3).index)) & (~mls_data['STYLE_CODE'].isna()),'STYLE_CODE'] = 'OTHER'


    #PARKING_TYPE cleaning
    mls_data.loc[(mls_data['PARKING_TYPE'].str.contains('UNFINISHED')) & (~mls_data['PARKING_TYPE'].isna()), 'PARKING_TYPE'] = 'UNFINISHED'
    mls_data.loc[(mls_data['PARKING_TYPE'].str.contains('CARPORT')) & (~mls_data['PARKING_TYPE'].isna()), 'PARKING_TYPE'] = 'CARPORT'
    mls_data.loc[(mls_data['PARKING_TYPE'].str.contains('FRAME')) & (~mls_data['PARKING_TYPE'].isna()), 'PARKING_TYPE'] = 'FRAME'
    mls_data.loc[(mls_data['PARKING_TYPE'].str.contains('BRICK')) & (~mls_data['PARKING_TYPE'].isna()), 'PARKING_TYPE'] = 'MASONRY'
    mls_data.loc[(mls_data['PARKING_TYPE'].str.contains('ATTACHED')) & (~mls_data['PARKING_TYPE'].isna()), 'PARKING_TYPE'] = 'ATTACHED'
    mls_data.loc[(mls_data['PARKING_TYPE'].str.contains('DETACHED')) & (~mls_data['PARKING_TYPE'].isna()), 'PARKING_TYPE'] = 'FINISHED'
    mls_data.loc[(mls_data['PARKING_TYPE'].str.contains('BASEMENT')) & (~mls_data['PARKING_TYPE'].isna()), 'PARKING_TYPE'] = 'UNFINISHED'

    #BASEMENT_TYPE_CODE cleaning
    mls_data.loc[(mls_data['BASEMENT_TYPE_CODE'].str.contains('PARTIAL')) & (~mls_data['BASEMENT_TYPE_CODE'].isna()), 'BASEMENT_TYPE_CODE'] = 'PARTIAL'
    mls_data.loc[(mls_data['BASEMENT_TYPE_CODE'].str.contains('UNFINISHED')) & (~mls_data['BASEMENT_TYPE_CODE'].isna()), 'BASEMENT_TYPE_CODE'] = 'PARTIAL'
    mls_data.loc[(mls_data['BASEMENT_TYPE_CODE'].str.contains('FINISHED')) & (~mls_data['BASEMENT_TYPE_CODE'].isna()), 'BASEMENT_TYPE_CODE'] = 'FULL'
    mls_data.loc[~mls_data['BASEMENT_TYPE_CODE'].isin(['CRAWL-SPACE', 'FULL', 'PARTIAL']) & (~mls_data['BASEMENT_TYPE_CODE'].isna()),'BASEMENT_TYPE_CODE'] = 'OTHER'

    mls_data = mls_data.convert_dtypes()

    return mls_data

# %%
#Clean Data
def merge_data(clean_sales_data, clean_mls_data):
    '''combines and cleans mls + sales data'''
    model_data = pd.merge(
        clean_sales_data, clean_mls_data, on='URL', how='left')
    model_data = model_data.replace(',', '', regex=True)
    model_data.drop_duplicates(inplace=True, ignore_index=True)
    # sqft_house: if SQUARE FEET is null then max(sqFtFinished, totalSqFt)
    model_data['SQUARE FEET'].fillna(model_data[['SQFTFINISHED','TOTALSQFT']].max(axis=1), inplace=True)
    model_data.drop(['SQFTFINISHED','TOTALSQFT'], axis=1, inplace=True)
    # sqft_lot: if LOT SIZE is null then lotSqFt
    model_data['LOT SIZE'].fillna(model_data[['LOTSQFT']].max(axis=1), inplace=True)
    model_data.drop(['LOTSQFT'], axis=1, inplace=True)
    # baths: if BATHS in null, FULL_BATHS+0.5*HALF_BATHS
    model_data['BATHS'].fillna(model_data['FULL_BATHS']+0.5*model_data['HALF_BATHS'], inplace=True)
    model_data.drop(['FULL_BATHS','HALF_BATHS'], axis=1, inplace=True)
    model_data.drop(['STATE OR PROVINCE', 'ZIP OR POSTAL CODE','CITY', 'URL', 'SOLD DATE'], axis=1, inplace=True)
    model_data = model_data.convert_dtypes()

    return model_data

# %%
