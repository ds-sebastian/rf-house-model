#%%
from dash.dependencies import Output, Input, State
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import datetime as dt
import json
import requests
from geopy.geocoders import Nominatim
import re
from data.functions.selenium_funcs import RedfinScraper
from data.functions.cleaning import mls_clean, merge_data
from datetime import date
from time import mktime
from app import app, server
from joblib import load
from data.functions.ml_functions import get_feature_names, get_feature_out, get_ct_feature_names
import pandas as pd
import numpy as np
#%%
now = dt.datetime.now()

table_header_width = 3

mls_data = dcc.Store(id='mls_data', storage_type='session')

property_type = dbc.FormGroup(
    [
        dbc.Label("Property Type", width=2),
        dbc.Col(
            dbc.RadioItems(
                options=[
                    {"label": "Single Family Home",
                        "value": 'SINGLE FAMILY RESIDENTIAL'},
                    {"label": "Townhome", "value": 'TOWNHOUSE'},
                    {"label": "Multi-Family", "value": 'MULTI-FAMILY'},
                    {"label": "Condo", "value": 'CONDO/CO-OP'},
                ],
                value='SINGLE FAMILY RESIDENTIAL',
                id="property_type",
                inline=True,
            ),
        )

    ],
    row=True
)

address = dbc.FormGroup(

    [dbc.Label("Address", width=table_header_width),
        dbc.Col(
        [dbc.Input(id="address-text", type="address",
                   placeholder="Enter Address"),
         dbc.Button("Search Redfin", id="submit-address-button",
                    color="primary"), ]
    )
    ],
    row=True
)

bed = dbc.FormGroup(
    [
        dbc.Label("Bedrooms", width=table_header_width),
        dbc.Col(
            dbc.Input(id="bedrooms", type="number", value=3, step=1),
        )
    ],
    inline=True,
    row=True
)


bath = dbc.FormGroup(
    [
        dbc.Label("Baths", width=table_header_width),
        dbc.Col(
            dbc.Input(id="baths", type="number", value=2, step=0.5),
        )
    ],
    row=True
)

sqft = dbc.FormGroup(
    [
        dbc.Label("Square Feet", width=table_header_width),
        dbc.Col(
            dbc.Input(id="sqft", type="number", value=1800,
                      min=0, max=50000, step=1),
        )
    ],
    row=True
)

lot_size = dbc.FormGroup(
    [
        dbc.Label("Lot Size", width=table_header_width),
        dbc.Col(
            dbc.Input(id="lotsize", type="number",
                      value=2000, min=0, max=50000, step=1),
        )
    ],
    row=True
)

market_days = dbc.FormGroup(
    [
        dbc.Label("Days On Market", width=table_header_width),
        dbc.Col(
            dbc.Input(id="market_days", type="number", value=1,
                      min=0, max=50000, step=1),
        )
    ],
    row=True
)

year_built = dbc.FormGroup(
    [
        dbc.Label("Year Built", width=table_header_width),
        dbc.Col(
            dbc.Input(id="year_built", type="number", value=2000,
                      min=1900, max=2050, step=1),
        )
    ],
    row=True
)

hoa_month = dbc.FormGroup(
    [
        dbc.Label("HOA/Month", width=table_header_width),
        dbc.Col(
            dbc.InputGroup(
                [
                    dbc.InputGroupAddon("$", addon_type="prepend"),
                    dbc.Input(id="hoa_month",
                              value=30, type="number"),
                    dbc.InputGroupAddon(".00", addon_type="append"),
                ],
            )
        )
    ],
    row=True
)

sales_form = dbc.Form([property_type, address, bed, bath,
                      sqft, lot_size, market_days, year_built, hoa_month])

prediction = html.Div([
    html.Div('''Sales Price Prediction:'''),
    html.H2(children=[
            html.Div(id='prediction_text', style={'display': 'inline'}),

            ], className="display-3")

])


@app.callback(Output('mls_data', 'data'),
              Input('submit-address-button', 'n_clicks'),
              State('address-text', 'value'))
def update_data_pull(n_clicks, query):

    if n_clicks is not None:

        geolocator = Nominatim(user_agent="address converter")
        user_agent_header = {
            'user-agent': 'redfin'
        }
        response = requests.get(
            'https://redfin.com/stingray/do/location-autocomplete', params={'location': query, 'v': 2}, headers=user_agent_header)
        response = json.loads(response.text[4:])
        geo = geolocator.geocode(query)
        latitude = geo.raw.get("lat")
        longitude = geo.raw.get("lon")

        try:
            url = response['payload']['exactMatch']['url']
        except:
            # First Row of search results
            url = response['payload']['sections'][0]['rows'][0]['url']

        property_id = re.search('([^\/]+$)', url)[0]
        api_url = 'https://www.redfin.com/stingray/api/home/details/belowTheFold?propertyId=' + \
            property_id+'&accessLevel=1&pageType=3'
        response = requests.get(api_url, headers=user_agent_header)
        response.raise_for_status()
        text = response.text

        clean_mls_data = mls_clean(RedfinScraper.mls_parse(url, text))

        clean_mls_data.insert(0, 'LONGITUDE', [longitude])
        clean_mls_data.insert(0, 'LATITUDE', [latitude])

        clean_mls_data = clean_mls_data.to_json(
            orient='split')

    else:
        clean_mls_data = {}

    return clean_mls_data


@app.callback(Output('prediction_text', 'children'),
              [
              Input('property_type', 'value'),
              Input('bedrooms', 'value'),
              Input('baths', 'value'),
              Input('sqft', 'value'),
              Input('lotsize', 'value'),
              Input('market_days', 'value'),
              Input('year_built', 'value'),
              Input('hoa_month', 'value'),
              Input('mls_data', 'data'),
              ])
def create_dataset(property_type, bed, bath, sqft, lot_size, market_days, year_built, hoa_month, mls_data):

    mls_data = pd.read_json(mls_data, orient='split').convert_dtypes()

    today = date.today()
    unixtime = mktime(today.timetuple())

    try:
        age = (today.year - year_built)
    except:
        age = 0

    sales_data = pd.DataFrame(
        {'SOLD DATE': '',
         'PROPERTY TYPE': property_type,
         'CITY': '',
         'STATE OR PROVINCE': '',
         'ZIP OR POSTAL CODE': '',
         'BEDS': bed,
         'BATHS': bath,
         'SQUARE FEET': sqft,
         'LOT SIZE': lot_size,
         'YEAR BUILT': year_built,
         'DAYS ON MARKET':  market_days,
         'HOA/MONTH': hoa_month,
         'URL': mls_data['URL'].iloc[0],
         'LATITUDE': mls_data['LATITUDE'].iloc[0],
         'LONGITUDE': mls_data['LONGITUDE'].iloc[0],
         'DATE SOLD_UNIX': unixtime,
         'AGE': age}, index=[0]).convert_dtypes()

    mls_data.drop(['LATITUDE', 'LONGITUDE'], inplace=True, axis=1)

    X = merge_data(sales_data, mls_data)

    dtypemap = {'PROPERTY TYPE': 'O', 'BEDS': 'float64', 'BATHS': 'float64', 'SQUARE FEET': 'float64',
                'LOT SIZE': 'float64', 'YEAR BUILT': 'float64', 'DAYS ON MARKET': 'float64',
                'HOA/MONTH': 'float64', 'LATITUDE': 'float64', 'LONGITUDE': 'float64',
                'DATE SOLD_UNIX': 'float64', 'AGE': 'float64', 'NUMSTORIES': 'float64',
                'YEARRENOVATED': 'float64', 'AIR_CONDITIONING_CODE': 'O', 'FUEL_CODE': 'O',
                'HEATING_TYPE_CODE': 'O', 'EXTERIOR_WALL_CODE': 'O', 'ROOF_TYPE_CODE': 'O',
                'ASSESSED_YEAR': 'float64', 'LIVING_SQUARE_FEET': 'float64',
                'BUILDING_SQUARE_FEET': 'float64', 'GARAGE_PARKING_SQUARE_FEET': 'float64',
                'GARAGE_CODE': 'O', 'NUMBER_OF_BUILDINGS': 'float64', 'PARENTRATING': 'float64',
                'SCHOOLDISANCE': 'float64', 'SERVESHOME': 'float64', 'NUMBEROFSTUDENTS': 'float64',
                'SCHOOL_SCORE': 'float64', 'STUDENT_TEACHER_RATIO': 'float64', 'REVIEW_NUMS': 'float64',
                'TAXABLELANDVALUE': 'float64', 'TAXABLEIMPROVEMENTVALUE': 'float64', 'ROLLYEAR': 'float64', 'TAXESDUE': 'float64'}

    X = X.astype(dtypemap).replace({pd.NA: np.nan})

    preprocessor = load('preprocessor.joblib')
    model = load('model.joblib')

    predict = preprocessor.transform(X)

    try:
        predict = pd.DataFrame(
            predict, columns=get_ct_feature_names(preprocessor))
    except:
        predict = pd.DataFrame.sparse.from_spmatrix(
            predict, columns=get_ct_feature_names(preprocessor))

    final_prediction = np.exp(model.predict(predict))
    final_prediction = "${:,.2f}".format(final_prediction[0])

    return [final_prediction]


layout = html.Div(

    dbc.Row(
        [html.Div(children=[
            html.H2('''Ral-AI House Predictor\n'''),
            html.Div(
                '''Search Redfin to get the latest MLS, School, and Location data. Fill in the rest of the inputs manually\n'''),
            sales_form,
            mls_data,
            prediction],
        )
        ]
    )
)
