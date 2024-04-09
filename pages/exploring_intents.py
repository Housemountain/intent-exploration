import json

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State
from dash import Input, Output, State, ALL, callback, ctx
import plotly.express as px

# initialize app
dash.register_page(__name__, path="/exploring_intent")
df_test_data = pd.DataFrame(
    {
        "x": [10, 1, 1, 0],
        "y": [1, 1, 0, -10],
        "model": ["cross", "cross", "ensemble", "ensemble"],
        'playlist_name': ["a", "b", "a", "b"],
        'intent': [0, 1, 0, 1],
        'intent_name': ['bla', 'bla bla', 'bbb', 'beew'],
        'score': [0.1, 0.2, 0.3, 0.4],
        'tracks': ["i", "ii", "iii", "iv"],
        'sim_playlists': ["1", "2", "3", "4"]
    })

layout = dbc.Container(
    [
        html.Br(),
        html.Br(),
        dbc.Row(html.H1("Intent exploration of Playlists", style={'textAlign': 'center'})),
        html.Br(),
        html.Br(),
        dbc.Row(dbc.RadioItems(
            id="choice_model",
            className="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-primary",
            labelCheckedClassName="active",
            options=[
                {"label": "Cross Encoder", "value": 0},
                {"label": "Ensemble", "value": 1}
            ],
            value=1,
            style={"width": "100%",
                   "display": 'flex',
                   "flex-direction": 'row',
                   'justify-content':
                       'center',
                   'justify': 'center',
                   'align-items': 'center'},
        ), ),
        dbc.Row([
            dbc.Col(dcc.Graph(id="intent_exploring_graph", style={'width': '100%', 'height': '80vh'}), width=10),
            dbc.Col([
                html.H4("Selected Playlist Profile", style={'textAlign': 'center'}),
                html.Br(),
                dcc.Markdown(id="selected_playlist_info")
            ], width=2)
        ], justify="center")
    ]
)


def make_fig(df, model):
    model_to_name = {0: "cross", 1: "ensemble"}
    df_ = df[df['model'] == model_to_name[model]]

    print("Scatter...")
    fig = px.scatter(df_,
                     x="x",
                     y="y",
                     color="intent",
                     hover_data=['playlist_name'],
                     color_continuous_scale=px.colors.sequential.Viridis,
                     size_max=12)

    # fig.update_layout(
    #    autosize=False,
    #    width=1600,
    #    height=800)

    return fig


@callback(
    [Output("intent_exploring_graph", "figure"), ],
    [
        Input("choice_model", "value"),
    ]
)
def on_playlist_name_changed(model_name):
    fig = make_fig(df_test_data, model_name)
    return fig,


@callback(
    Output('selected_playlist_info', 'children'),
    Input('intent_exploring_graph', 'clickData'))
def display_click_data(clickData):
    if clickData is None:
        return "No playlist selected."
    print(clickData['points'][0].keys())
    playlist_name = clickData['points'][0]["customdata"][0]

    df_p = df_test_data[df_test_data['playlist_name'] == playlist_name]

    info = f"""
    Playlist Name: {playlist_name}

    Intent: {df_p['intent_name'].iloc[0]}

    Similarity Score: {df_p['score'].iloc[0]}

    First tracks: {df_p['tracks'].iloc[0]}

    Similar playlists: {df_p['sim_playlists'].iloc[0]}
    """

    return info
