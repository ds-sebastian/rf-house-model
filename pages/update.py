#%%

# Imports from 3rd party libraries
import dash
from dash.dependencies import Output,Input
import dash_table as dct
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.exceptions import PreventUpdate
import dash_html_components as html

from app import app, server
from data.updater import RedfinData

#%%

updater = RedfinData()

spinner = dcc.Loading(children=html.Div(id="loading"), color="info", type="default", fullscreen=True,)

buttons = dbc.Col(
    [   
        html.H3("Update"),
        dbc.Button("Sales Data", id="sales-button",
                    color="primary", className='mr-2', n_clicks=0),
        dbc.Button("MLS Data", id="mls-button",
                    color="primary", className='mr-2', n_clicks=0),
        dbc.Button("Model Data", id="model-button",
                    color="primary", className='mr-2', n_clicks=0),
        dbc.Button("All Data", id="all-button",
                    color="warning", className='mr-2', n_clicks=0),      
    ],
)

@app.callback(Output("loading", "children")
            , [Input("sales-button", "n_clicks"),
            Input("mls-button", "n_clicks"),
            Input("model-button", "n_clicks"),
            Input("all-button", "n_clicks")]
            )
def update(sales, mls, model, all):
    ctx = dash.callback_context

    if ctx.triggered[0]['prop_id'] == 'sales-button.n_clicks':
        updater.update_sales_data()
        updater.exit_browser()
    elif ctx.triggered[0]['prop_id'] == 'mls-button.n_clicks':
        updater.update_mls_data()
        updater.exit_browser()
    
    elif ctx.triggered[0]['prop_id'] == 'model-button.n_clicks':
        updater.update_model_data()
    
    elif ctx.triggered[0]['prop_id'] == 'all-button.n_clicks':
        updater.complete_update()
        updater.exit_browser()

    else:
        PreventUpdate

    return ctx.triggered[0]['prop_id']


layout = html.Div([spinner, buttons])