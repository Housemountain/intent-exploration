import json

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from click import style
from dash import Dash, dcc, html, Input, Output, State, dash_table, no_update
from dash import Input, Output, State, ALL, callback, ctx
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.cyextension.resultproxy import rowproxy_reconstructor

# initialize app
dash.register_page(__name__, path="/exploring_intent")

PYTHONANYWHERE_PATH = '/home/ismir20241B0D/intent-exploration'
# PYTHONANYWHERE_PATH = './'

df_intent = pd.read_json(f"{PYTHONANYWHERE_PATH}/data/new_cluster_data.json")
df_playlists = pd.read_json(f"{PYTHONANYWHERE_PATH}/data/playlist_data_scored.json")
df_playlists = df_playlists[(df_playlists['scored'] == True) & (df_playlists['scaled'] == True)]
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
        dbc.Row(html.H1("Investigating Music Listening Intents in User-generated Playlist Titles",
                        style={'textAlign': 'center'})),
        html.Br(),
        dcc.Markdown(
            """People listen to music for various reasons, for which the underlying listening intents can be broken down into concrete music listening functions. %Music listening functions can be defined as reasons people listen to music. 
These have been identified through empirical studies in music psychology, commonly using interviews and surveys. 
In this paper, we take a data-driven approach that adopts data augmentation techniques via large language models, pre-trained Sentence Transformers and Cross Encoder, and graph-based clustering to explore the occurring music listening intents in user-generated playlists by comparing the title to the listening intents.
For this purpose, we first investigate whether 129 established listening functions, previously identified through a survey, can be meaningfully clustered into larger listening intents. The resulting clusters are evaluated through a pilot survey.
Given the encouraging results of the evaluation of the computed clusters (92% of clusters judged consistent by participants), we match the playlist titles to the listening functions, and compute the similarity score for each intent cluster. Based on the similarity score vector, we determine the intent of a playlist, and investigate measures to ensure a playlist can be assigned to an intent or not. Further, we retrieve similar playlists on basis of this similarity score vector.
We present a dashboard to explore playlists in an intent space to find similar playlists on the basis of intent."""),
        html.Br(),
        html.Br(),
        html.H2("Explore Intent Clusters", style={'textAlign': 'center'}),

        dcc.Markdown("""In this section, you can explore our intent clusters. Select the main music function and explore the assigned playlists, 
        resulted names from the user study and much more. 
        Start the exploration by selecting the main music function."""),
        html.Br(),
        dbc.Row(dbc.Row(dbc.Col(dcc.Dropdown(
            id="choice_intent",
            options=list(set(list(df_intent['main_question']))),
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
        html.Br(),

        html.H2("Browsing through Playlists by Intent", style={'textAlign': 'center'}),
        dcc.Markdown("""In this section, you can select a intent-based query and explore its intent classification and playlists suggested by our 
        browsing system by our intent-based vector. Select different models to explore how the intent and similar playlists change.
        Start this exploration by selecting a query and a model."""),
        html.Br(),
        html.H5("Select query text for browsing", style={'textAlign': 'center'}),
        dbc.Row(dbc.Row(dbc.Col(dcc.Dropdown(
            id="choice_text",
            options=list(set(list(df_playlists[df_playlists['query'] == True]['playlist']))),
        )),
            style={"width": "50%",
                   "display": 'flex',
                   "flex-direction": 'row',
                   'justify-content':
                       'center',
                   'justify': 'center',
                   'align-items': 'center'}, ), justify="center", ),
        html.Br(),
        dbc.Row(id="row_model_selection", justify='center'),
        html.Br(),
        dbc.Row(id='row_intent', justify="center"),
        html.Br(),
        dbc.Row(id='row_most_sim', justify="center"),
        html.Br(),
        dbc.Row(id='row_graph', justify="center")
    ]
)


def make_fig(df_p):
    print("Scatter...")
    fig = px.scatter(df_p,
                     x="x",
                     y="y",
                     color="main function",
                     hover_data=['playlist'],
                     symbol="query",
                     size='score',
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
    [Output('row_model_selection', 'children')],
    [
        Input("choice_text", "value"),
    ]
)
def on_query_selected(choice_text):
    if choice_text is None or len(choice_text) <= 0:
        return [html.H5("Choose a query to get more information about the intent.",
                        style={'textAlign': 'center'})],
    model_selection = [html.H5("Select model for computing similarity to clusters", style={'textAlign': 'center'}),
                       dbc.Row(
                           dcc.Tabs(id="choice_model", value="0", children=[
                               dcc.Tab(label="Cross Encoder (stsb-roberta-base)", value="0"),
                               dcc.Tab(label="Sentence Transformer (all-mpnet-base-v2)", value="1"),
                               dcc.Tab(label="Sentence Transformer (quora-distilbert-base)", value="2"),
                               dcc.Tab(label="Sentence Transformer (all-MiniLM-L12-v2)", value="3"),
                               dcc.Tab(label="Ensemble", value="4")
                           ],
                                    style={"width": "100%",
                                           "display": 'flex',
                                           "flex-direction": 'row',
                                           'justify-content':
                                               'center',
                                           'justify': 'center',
                                           'align-items': 'center'},
                                    ), ),
                       html.Br(), ]

    return model_selection,


@callback([Output("row_intent", "children"), Output("row_most_sim", "children"), Output("row_graph", "children")],
          [Input("choice_text", "value"), Input("choice_model", "value")])
def change_information(current_query, model_name):
    df_model = df_playlists[(df_playlists['model'] == model_to_name[int(model_name)])]

    df_query = df_model[(df_model['playlist'] == current_query)].iloc[0]

    q_intent_vec = np.array(df_query['intent_vec'])
    df_p = df_model[df_model['query'] == False]
    other = np.array([np.array(i) for i in df_p['intent_vec']])

    sim = cosine_similarity(q_intent_vec.reshape(1, -1), other)[0]
    df_p['sim'] = [f'{s:.2f}' for s in sim]

    df_p = df_p.sort_values(by=['sim'], ascending=False)

    playlists = list(df_p.head(100)['playlist']) + [current_query]
    df_to_viz = df_model[df_model['playlist'].isin(playlists)]

    fig = make_fig(df_to_viz)

    row_intent = [
        html.H5("Intent of Query", style={'textAlign': 'center'}),
        dcc.Markdown(f"""The intent of this query has the main listening function: **{df_query['main function']}**""",
                     style={'textAlign': 'center'})
    ]

    def get_url(row):
        return f"""<a
        href = "https://open.spotify.com/search/{row}/playlists"
        target = "_blank" >{row}</a>"""

    df_p['Playlist'] = df_p['playlist'].apply(get_url)
    df_p['Similarity'] = df_p['sim']
    df_p['Main Function'] = df_p['main function']

    row_most_sim = [
        html.H5("Top 10 most similar Playlists to Query", style={'textAlign': 'center'}),
        dbc.Table.from_dataframe(df_p[['Playlist', 'Similarity', 'Main Function']].head(10), striped=True, bordered=True,
                                 hover=True)
    ]

    row_graph = [html.H5(f"Query visualized in lower space with Top 100 most similar Playlists with Intent",
                         style={'textAlign': 'center'}),
                 dcc.Graph(id="intent_exploring_graph", figure=fig, style={'width': '100%', 'height': '80vh'})]

    return row_intent, row_most_sim, row_graph


@callback(
    [Output('selected_playlist_info', 'children'),
     Output('playlist_embed_01', 'href')],
    [
        Input("choice_model", "value"),
        Input("choice_text", "value"),
        Input('intent_exploring_graph', 'clickData')
    ])
def display_click_data(model, query, clickData):
    if clickData is None:
        return "Please click on a playlist in the graph to get more information."

    df_coords_ = df_playlists[
        (df_playlists['model'] == model_to_name[model])]

    print(clickData['points'][0].keys())
    playlist_name = clickData['points'][0]["customdata"][0]

    df_p_c = df_coords_[df_coords_['playlist'] == playlist_name]

    info = f"""
    Playlist Name: {playlist_name}

    Intent: {df_p_c['intent'].iloc[0]}

    Similarity Score: {df_p_c['sim_score'].iloc[0]}

    """

    url = 'https://open.spotify.com/search/' + playlist_name + '/playlists'

    return info, url


@callback(
    [
        Output('container_intent_information', 'children'),
        Output("row_intent_names", 'children'),
        Output("container_listening", 'children'),
        Output("container_intent_top_playlists", 'children')
    ],
    [
        Input("choice_intent", "value")
    ])
def display_intent_data(mf):
    if mf is None or len(mf) <= 0:
        return [html.H5("Choose a music listening function to get more information about the intent.",
                        style={'textAlign': 'center'})], [], [], []

    df_mf = df_intent[df_intent['main_question'] == mf].iloc[0]

    c_idx = df_mf['cluster']

    playlist_data = {t: [] for t in idx_to_title.values()}
    for idx, model in model_to_name.items():  # idx_to_title.items():
        df_model = t_to_c_score[(t_to_c_score['model'] == model) & (t_to_c_score['query'] == False)].sort_values(
            by=[f"c_{c_idx}"], ascending=False).head()
        playlists = [f"{p} ({s:.2f})" for p, s in zip(df_model['playlist'], df_model[f'c_{c_idx}'])]
        playlist_data[idx_to_title[idx]] = playlists

    for k, v in playlist_data.items():
        print(k, len(v))
    df_playlist_data = pd.DataFrame(playlist_data)
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
    return [
        html.Br(),
        html.Br(),
        html.H5("Intent Cluster",
                style={'textAlign': 'center'}), dcc.Markdown("""The following table shows all music functions assigned to this cluster. 
                        The score indicates how often the music function was selected to be in the same intent cluster as the main function. 
                        The score ranges from 0 to 1, where 1 indicates that all 10 people have selected this music function OR the music function is the main music function. 
                        The main music function has been selected to be the function with the highest mean similarity to all music functions in the cluster."""),
        html.Br(),
        dbc.Table.from_dataframe(new_df, striped=True, bordered=True, hover=True)], all_cols, [dcc.Markdown(
        f"""People listen to this intent on average: **{np.array(df_mf['listen']).mean()}** (0 == 'Never', 10 == 'Daily')""",
        style={'textAlign': 'center'})], [
        html.Br(),
        html.H5("Top 5 Playlists with highest score for this intent",
                style={'textAlign': 'center'}), dcc.Markdown(
            """The following table shows the top 5 playlists with the classified intent sorted by score for the used models to compute the intent vector."""),
        html.Br(),
        dbc.Table.from_dataframe(df_playlist_data, striped=True, bordered=True, hover=True)
    ]

#
