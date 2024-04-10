import json

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State
from dash import Input, Output, State, ALL, callback, ctx
import plotly.express as px

# initialize app
dash.register_page(__name__, path="/exploring_intent")

PYTHONANYWHERE_PATH = '/home/ismir20241B0D/intent-exploration'

df_coords = pd.read_json(f"{PYTHONANYWHERE_PATH}/data/coords.json")
df_coords['intent'] = df_coords['cluster']
df_sim_playlists = pd.read_json(f"{PYTHONANYWHERE_PATH}/data/most_sim_playlists.json")

df_coords_ = None
df_sim_ = None

model_to_name = {0: "cross", 1: "ensemble"}
augmented_to_name = {0: 'original', 1: 'augmented'}

layout = dbc.Container(
    [
        html.Br(),
        html.Br(),
        dbc.Row(html.H1("Investigating Music Listening Intents in User-generated Playlist Titles", style={'textAlign': 'center'})),
        html.Br(),
        dcc.Markdown(
            """People listen to music for various reasons, for which the underlying listening intents can be broken down into concrete music listening functions. %Music listening functions can be defined as reasons people listen to music. 
These have been identified through empirical studies in music psychology, commonly using interviews and surveys. 
In this paper, we take a data-driven approach that adopts data augmentation techniques via large language models, pre-trained Sentence Transformers and Cross Encoder, and graph-based clustering to explore the occurring music listening intents in user-generated playlists by comparing the title to the listening intents.
For this purpose, we first investigate whether 129 established listening functions, previously identified through a survey, can be meaningfully clustered into larger listening intents. The resulting clusters are evaluated through a pilot survey.
Given the encouraging results of the evaluation of the computed clusters (92\% of clusters judged consistent by participants), we match the playlist titles to the listening functions, and compute the similarity score for each intent cluster. Based on the similarity score vector, we determine the intent of a playlist, and investigate measures to ensure a playlist can be assigned to an intent or not. Further, we retrieve similar playlists on basis of this similarity score vector.
We present a dashboard to explore playlists in an intent space to find similar playlists on the basis of intent."""),
        html.Br(),
        html.H5("Select model for computing similarity to clusters", style={'textAlign': 'center'}),
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
        html.Br(),
        html.H5("Select playlist text used for computation", style={'textAlign': 'center'}),
        dbc.Row(dbc.RadioItems(
            id="choice_text",
            className="btn-group",
            inputClassName="btn-check",
            labelClassName="btn btn-outline-primary",
            labelCheckedClassName="active",
            options=[
                {"label": "Original Playlist Title", "value": 0},
                {"label": "Augmented Playlist Title", "value": 1}
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
                html.H5("Playlist Profile", style={'textAlign': 'center'}),
                html.Br(),
                dcc.Markdown(id="selected_playlist_info")
            ], width=2)
        ], justify="center")
    ]
)


def make_fig(model, augmented):
    df_coords_ = df_coords[(df_coords['model'] == model_to_name[model]) & (df_coords['sim_type'] == augmented_to_name[augmented])]

    print("Scatter...")
    fig = px.scatter(df_coords_,
                     x="x",
                     y="y",
                     color="intent",
                     hover_data=['playlist'],
                     color_continuous_scale=px.colors.sequential.Viridis,
                     size_max=12)

    fig.update_layout(
        yaxis_title=None,
        xaxis_title=None)
    #    autosize=False,
    #    width=1600,
    #    height=800)

    return fig


@callback(
    [Output("intent_exploring_graph", "figure"), ],
    [
        Input("choice_model", "value"),
        Input("choice_text", "value"),
    ]
)
def on_playlist_name_changed(model_name, choice_text):
    fig = make_fig(model_name, choice_text)
    return fig,


@callback(
    Output('selected_playlist_info', 'children'),
    [
        Input("choice_model", "value"),
        Input("choice_text", "value"),
        Input('intent_exploring_graph', 'clickData')
    ])
def display_click_data(model, text, clickData):
    if clickData is None:
        return "Please click on a playlist in the graph to get more information."

    df_coords_ = df_coords[
        (df_coords['model'] == model_to_name[model]) & (df_coords['sim_type'] == augmented_to_name[text])]
    df_sim_ = df_sim_playlists[(df_sim_playlists['model'] == model_to_name[model]) & (
                df_sim_playlists['sim_type'] == augmented_to_name[text])]

    print(clickData['points'][0].keys())
    playlist_name = clickData['points'][0]["customdata"][0]

    df_p = df_sim_[df_sim_['playlist'] == playlist_name]
    df_p_c = df_coords_[df_coords_['playlist'] == playlist_name]

    info = f"""
    Playlist Name: {playlist_name}

    Intent: {df_p_c['intent'].iloc[0]}

    Similarity Score: {df_p_c['sim_score'].iloc[0]}

    Similar playlists: {",".join(df_p['most_sim_playlists'].iloc[0])}
    """

    return info
