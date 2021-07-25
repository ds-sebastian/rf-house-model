# Imports from 3rd party libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

# Imports from this application
from app import app

# 2 column layout. 1st column width = 4/12
# https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout
column1 = dbc.Col(
    [
        dcc.Markdown(
            """
        
            ## Raleigh Area Real Estate Model
            \n
            This webapp uses available Redfin data to train and maintain a Machine Learning model
            predicting the sales price of homes in the Triangle Area
            \n
            """
        ),
        dcc.Link(dbc.Button('Predictions', color='primary'), href='/predictions'),
        
        dcc.Link(dbc.Button('Update Data', color='secondary'), href='/update'),
        
        #dcc.Link(dbc.Button('Log New Data', color='warning'), href='/logdata')
    ],
    md=4,
)

image = html.Div(
        children=html.Img(
            src="https://assets.simpleviewinc.com/simpleview/image/upload/c_fill,h_571,q_80,w_1603/v1/clients/raleigh/165_3_0042_jpeg_a32ab91e-8245-42f1-baa2-36a6f2a54dbb.jpg",
            style={
                'maxWidth': '70%',
                'maxHeight': '70%',
                'marginLeft': 'auto',
                'marginRight': 'auto'
            }
        ),
)


column2 = dbc.Col(
    [
        image,
    ]
)

layout = dbc.Row([column1, column2])