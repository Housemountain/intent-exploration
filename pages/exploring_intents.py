import json

import dash
import dash_bootstrap_components as dbc
import dash_player
import numpy as np
import pandas as pd
from click import style
from dash import Dash, dcc, html, Input, Output, State, dash_table, no_update
from dash import Input, Output, State, ALL, callback, ctx
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.cyextension.resultproxy import rowproxy_reconstructor

import dash_player as dp

# initialize app
dash.register_page(__name__, path="/exploring_intent")

PYTHONANYWHERE_PATH = '/home/ismir20241B0D/intent-exploration'
PYTHONANYWHERE_PATH = '/home/eximb/mysite/intent-exploration'
PYTHONANYWHERE_PATH = './'

df_intent = pd.read_json(f"{PYTHONANYWHERE_PATH}/data/new_cluster_data.json")

df_songs = pd.read_json(f"{PYTHONANYWHERE_PATH}/data/playlist_songs.json")

df_playlists = pd.read_json(f"{PYTHONANYWHERE_PATH}/data/playlist_data_scored.json")
df_playlists = df_playlists[(df_playlists['scored'] == True) & (df_playlists['scaled'] == True)]

title = []

i_to_title = {}

for i, t in zip(df_intent['cluster'], df_intent['title']):
    i_to_title[i] = t

for i in df_playlists['intent']:
    title.append(i_to_title[i])

df_playlists['intent_title'] = title

t_to_c_score = {"playlist": [], "model": [], "query": []}

for t, m, q, i_vec in zip(df_playlists['playlist'], df_playlists['model'], df_playlists['query'],
                          df_playlists['intent_vec']):
    t_to_c_score["playlist"].append(t)
    t_to_c_score["model"].append(m)
    t_to_c_score["query"].append(q)

    for idx, s in enumerate(i_vec):
        if f"c_{idx}" not in t_to_c_score:
            t_to_c_score[f"c_{idx}"] = []
        t_to_c_score[f"c_{idx}"].append(s)

t_to_c_score = pd.DataFrame(t_to_c_score)

model_to_name = {0: "stsb-roberta-base",
                 1: "all-mpnet-base-v2",
                 2: "quora-distilbert-base",
                 3: "all-MiniLM-L12-v2",
                 4: "ensemble"}

idx_to_title = {
    0: 'stsb-roberta-base (Cross Encoder)',
    1: "all-mpnet-base-v2 (Sentence Transformer)",
    2: "quora-distilbert-base (Sentence Transformer)",
    3: "all-MiniLM-L12-v2 (Sentence Transformer)",
    4: "Ensemble"
}

layout = dbc.Container(
    [
        html.Br(),
        html.Br(),
        dbc.Row(html.H1("ExIM: Exploring Intent of Music Listening for Retrieving User-generated Playlists",
                        style={'textAlign': 'center'})),
        html.Br(),
        dcc.Markdown(
            """Music psychology has identified various reasons why people listen
to music, based on empirical evidence from interviews and sur-
veys. In this paper, we take a data-driven approach that adopts
both pre-trained Sentence Transformers and Cross Encoder, as
well as graph-based clustering to first determine music listening
intents and then explore user-generated playlists by comparing the
title to the listening intents. For this purpose, we first investigated
whether 129 established listening functions, previously identified by
Schäfer et al. (2013), could be meaningfully clustered into broader
listening intents. While Schäfer et al. (2013) introduced three broad
dimensions of music listening, this work aimed to identify smaller,
context-specific intents to capture more nuanced intents. The re-
sulting clusters were then evaluated through a first survey to select
the clusters of the best performing model. In a second survey, music
listening intent clusters were explored in more detail to obtain a
deeper understanding of their significance for music retrieval and
recommendation. Lastly, the playlist selection per intent and char-
acteristics of listening with intent were further explored through
a third survey. Given the encouraging results of the evaluation of
the computed clusters (92% of clusters judged consistent by partici-
pants) and the insight that most (> 50%) of the participants search
for playlists for a specific intent, we propose a browsing system
that categorizes playlists based on their intent and enables users
to explore similar playlists. Our approach is further visualized in a
dashboard to explore and browse through playlists in intent space."""),
        html.Br(),
        html.Br(),
        html.H2("Browsing by Intent", style={'textAlign': 'center'}),
        dcc.Markdown("""Here, you can search, explore and listen to playlists by intent. Select your intent and find playlists, or explore the whole intent space below. What is your intent of listening to music?"""),
        html.Br(),
        dbc.Row(dbc.Row(dbc.Col(dcc.Dropdown(
            id="choice_intent",
            options=list(set(list(df_intent['title']))),
        )),
            style={"width": "50%",
                   "display": 'flex',
                   "flex-direction": 'row',
                   'justify-content':
                       'center',
                   'justify': 'center',
                   'align-items': 'center'}, ), justify="center", ),

        html.Br(),
        dbc.Container(id="container_listening"),
        dbc.Row(id="row_intent_names", justify="center"),
        dbc.Container(id="container_intent_information"),
        dbc.Container(id="container_intent_top_playlists"),
        # What to show?
        # Selection of Main Music Function
        # Cluster Information with selection Scores
        html.Br(),
        html.Br(),
        dbc.Row(id='row_graph', justify="center"),
        html.Br(),
        html.Br(),
        dbc.Row(id='row_music', justify='center'),
    ]
)


def make_fig(df_p):
    print("Scatter...")
    fig = px.scatter(df_p,
                     x="x",
                     y="y",
                     color="model",
                     hover_data=['playlist', 'model'],
                     color_continuous_scale=px.colors.sequential.Viridis,
                     size_max=12)

    fig.update_layout(coloraxis_colorbar=dict(yanchor="top", y=1, x=-0.09,
                                              ticks="outside"))

    fig.update_layout(
        yaxis_title=None,
        xaxis_title=None)
    #    width=1600,
    #    height=800)

    return fig

@callback(
    [Output('row_music', 'children')],
    [
        Input('intent_exploring_graph', 'clickData')
    ])
def display_click_data(clickData):
    if clickData is None:
        return "Please click on a playlist in the graph to get more information."

    print(clickData)

    data = clickData['points'][0]['customdata']

    playlist = data[0]
    model = data[1]

    df_p = df_playlists[(df_playlists['playlist'] == playlist) & (df_playlists['model'] == model)].iloc[0]

    songs = df_songs[df_songs['playlist'] == df_p['playlist']].iloc[0]
    print(songs)

    audio_children = [
        html.H3(f"Explore the Playlist {playlist}", style={'textAlign': 'center'}),
        html.Br(),
        html.Br(),
        html.Br(),
        dbc.Row(children=[
            dbc.Col(children=[html.H5("Intent", style={'textAlign': 'center'}), html.H4(html.B(df_p['intent_title']), style={'textAlign': 'center'})]),
            dbc.Col(children=[html.H5("Evaluated By", style={'textAlign': 'center'}), html.H4(html.B(model), style={'textAlign': 'center'})]),
        ], justify="center"),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.H4("Listen to the 10 most occurring Songs", style={'textAlign': 'center'}),
        html.Br(),
        html.Br(),
    ]

    for s_url, s_data in zip(songs['songs'], songs['song_data']):
        audio_children.append(html.H5(s_data, style={'textAlign': 'center'}))
        audio_children.append(html.Br())
        audio_children.append(html.Br())
        audio_children.append(dash_player.DashPlayer(
            id="player",
            url=s_url,
            controls=True,
            width="40%",
            height="25px",))
        audio_children.append(html.Br())
        audio_children.append(html.Br())

    return audio_children,


@callback(
    [
        Output('container_intent_information', 'children'),
        Output("row_intent_names", 'children'),
        Output("container_listening", 'children'),
        Output("container_intent_top_playlists", 'children'),
        Output('row_graph', 'children')
    ],
    [
        Input("choice_intent", "value")
    ])
def display_intent_data(mf):
    if mf is None or len(mf) <= 0:
        return [html.H5("Choose your intent to browse through the playlists.",
                        style={'textAlign': 'center'})], [], [], []

    df_mf = df_intent[df_intent['title'] == mf].iloc[0]

    c_idx = df_mf['cluster']

    p_to_viz = []

    playlist_data = {t: [] for t in idx_to_title.values()}
    for idx, model in model_to_name.items():  # idx_to_title.items():
        df_model = t_to_c_score[(t_to_c_score['model'] == model) & (t_to_c_score['query'] == False)].sort_values(
            by=[f"c_{c_idx}"], ascending=False).head(100)
        playlists = [f"{p} ({s:.2f})" for p, s in zip(df_model['playlist'], df_model[f'c_{c_idx}'])]
        playlist_data[idx_to_title[idx]] = playlists

        p_to_viz.extend(list(df_model['playlist']))

    for k, v in playlist_data.items():
        print(k, len(v))
    df_playlist_data = pd.DataFrame(playlist_data).head()
    print(df_mf.keys())

    new_df = {"Music Listening Function": [f'{f} (Main)' if f == mf else f for f in list(df_mf['questions'])],
              "Score": list(df_mf['survey'])}
    new_df = pd.DataFrame(new_df).sort_values(by=["Score"], ascending=False)

    names = list(set([title.strip() for title in list(set(df_mf['given_names'])) if title is not None]))
    print(names)
    cols = []
    all_cols = [html.H5("Names given to this intent", style={'textAlign': 'center'}),
                dcc.Markdown(
                    """The names were given by the participants of our survey based on how they perceived the intent of the cluster of music listening functions.""",
                    style={'textAlign': 'center'})]
    for n in names:
        if n is not None and len(n) > 0:
            cols.append(dbc.Col(dbc.Card(n, body=True, style={'justify-content': 'center', 'justify': 'center',
                                                              'align-items': 'center'}), width=2))

        if len(cols) >= 5:
            all_cols.append(dbc.Row(cols, style={"width": "80%",
                                                 "display": 'flex',
                                                 "flex-direction": 'row',
                                                 'justify-content':
                                                     'center',
                                                 'justify': 'center',
                                                 'margin-top': '12px',
                                                 'margin-bottom': '12px',
                                                 'align-items': 'center'}))
            cols = []

    if len(cols) > 0:
        all_cols.append(dbc.Row(cols, style={"width": "80%",
                                             "display": 'flex',
                                             "flex-direction": 'row',
                                             'margin-top': '12px',
                                             'margin-bottom': '12px',
                                             'justify-content':
                                                 'center',
                                             'justify': 'center',
                                             'align-items': 'center'}))


    intent_cluster = [
        html.Br(),
        html.Br(),
        html.H5("Intent Cluster",
                style={'textAlign': 'center'}), dcc.Markdown("""The following table shows all music functions assigned to this cluster. 
                        The score indicates how often the music function was selected to be in the same intent cluster as the main function. 
                        The score ranges from 0 to 1, where 1 indicates that all 10 people have selected this music function OR the music function is the main music function. 
                        The main music function has been selected to be the function with the highest mean similarity to all music functions in the cluster."""),
        html.Br(),
        dbc.Table.from_dataframe(new_df, striped=True, bordered=True, hover=True)]

    listen = [dcc.Markdown(
        f"""People listen to this intent on average: **{np.array(df_mf['listen']).mean()}** (0 == 'Never', 10 == 'Daily')""",
        style={'textAlign': 'center'})]

    top_playlists = [
        html.Br(),
        html.H5("Top 5 Playlists with highest score for this intent",
                style={'textAlign': 'center'}), dcc.Markdown(
            """The following table shows the top 5 playlists with the classified intent sorted by score for the used models to compute the intent vector."""),
        html.Br(),
        dbc.Table.from_dataframe(df_playlist_data, striped=True, bordered=True, hover=True)
    ]

    fig = make_fig(df_playlists[(df_playlists['query'] == False) & (df_playlists['playlist'].isin(p_to_viz))])
    graph = [html.H5(f"Visualization of Top 100 Playlists with the Highest Score for the selected intent per model",
                         style={'textAlign': 'center'}),
             html.P("""Click on a playlist in the graph to listen to it and explore the playlist a bit more.""",  style={'textAlign': 'center'}),
                 dcc.Graph(id="intent_exploring_graph", figure=fig, style={'width': '100%', 'height': '80vh'})]

    return intent_cluster, all_cols, listen, top_playlists, graph

#
