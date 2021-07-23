#%%

# Imports from 3rd party libraries
from socket import AddressFamily
from dash.dependencies import Output, Input, State
import dash_table as dct
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import datetime as dt
import os
import json
import requests
from geopy.geocoders import Nominatim
import re
from data.functions.selenium_funcs import mls_parse
from data.functions.cleaning import mls_clean, merge_data
from datetime import date
from time import mktime
from app import app, server

import pandas as pd
#%%
now = dt.datetime.now()


mls_data = dcc.Store(id='mls_data', storage_type='session')
final_data = dcc.Store(id='final_data', storage_type='session')
#USE DCC STORE!!!!! SUPER BETTER
#https://dash-bootstrap-components.opensource.faculty.ai/docs/components/input/
#https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout

property_type = dbc.FormGroup(
    [
        dbc.Label("Choose Property Type", width=2),
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

    [dbc.Label("Address", width=2),
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
        dbc.Label("Bedrooms", width=2),
        dbc.Col(
            dbc.Input(id="bedrooms", type="number", value = 3, step=1),
        )
    ],
    inline=True,
    row=True
)


bath = dbc.FormGroup(
    [
        dbc.Label("Baths", width=2),
        dbc.Col(
            dbc.Input(id="baths", type="number", value = 2, step=0.5),
        )
    ],
    row=True
)

sqft = dbc.FormGroup(
    [
        dbc.Label("Square Feet", width=2),
        dbc.Col(
            dbc.Input(id="sqft", type="number", value=1800, min=0, max=50000, step=1),
        )
    ],
    row=True
)

lot_size = dbc.FormGroup(
    [
        dbc.Label("Lot Size", width=2),
        dbc.Col(
            dbc.Input(id="lotsize", type="number", value=2000, min=0, max=50000, step=1),
        )
    ],
    row=True
)

market_days = dbc.FormGroup(
    [
        dbc.Label("Days On Market", width=2),
        dbc.Col(
            dbc.Input(id="market_days", type="number", value = 1,
                      min=0, max=50000, step=1),
        )
    ],
    row=True
)

year_built = dbc.FormGroup(
    [
        dbc.Label("Year Built", width=2),
        dbc.Col(
            dbc.Input(id="year_built", type="number", value = 2000,
                      min=1900, max=2050, step=1),
        )
    ],
    row=True
)

hoa_month = dbc.FormGroup(
    [
        dbc.Label("HOA $ Per Month", width=2),
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

#inputs = html.Div(
#    [
#        sales_form,
#    ]
#)


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
        #geo = geolocator.geocode(query)
        latitude = 35 #geo.raw.get("lat")
        longitude = 35 #geo.raw.get("lon")

        try:
            url = response['payload']['exactMatch']['url']
        except:
            url = response['payload']['sections'][0]['rows'][0]['url'] #First Row of search results


        property_id = re.search('([^\/]+$)', url)[0]
        api_url = 'https://www.redfin.com/stingray/api/home/details/belowTheFold?propertyId=' + \
            property_id+'&accessLevel=1&pageType=3'
        response = requests.get(api_url, headers=user_agent_header)
        response.raise_for_status()
        text = response.text

        clean_mls_data = mls_clean(mls_parse(url, text))

        clean_mls_data.insert(0, 'LONGITUDE', [longitude])
        clean_mls_data.insert(0, 'LATITUDE', [latitude])

        
        #clean_mls_data.to_json() #TEMP

        clean_mls_data = clean_mls_data.to_json(orient='split')  #dict('records')[0]

    else:
        clean_mls_data = {}

    #clean_mls_data = clean_mls_data.applymap(str)

    #print(clean_mls_data)

    return clean_mls_data


@app.callback(Output('final_data', 'data'),
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

    mls_data = pd.read_json(mls_data, orient='split')

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
         'PRICE': 0,
         'BEDS': bed,
         'BATHS': bath,
         'SQUARE FEET': sqft,
         'LOT SIZE': lot_size,
         'YEAR BUILT': year_built,
         'DAYS ON MARKET':  market_days,
         'HOA/MONTH': hoa_month,
         'URL': '',
         'LATITUDE': mls_data['LATITUDE'].iloc[0],
         'LONGITUDE': mls_data['LONGITUDE'].iloc[0],
         'DATE_SOLD_UNIX': unixtime,
         'AGE': age}, index=[0])
    
    #print(mls_data)

    mls_data.drop(['LATITUDE', 'LONGITUDE'], inplace=True, axis = 1)

    finaldata = merge_data(sales_data, mls_data)
    #finaldata = finaldata.loc[0]
    print('/n')
    print(finaldata.columns)

    return finaldata.to_json(orient='split')


layout = html.Div([sales_form, mls_data, final_data])

# %%
