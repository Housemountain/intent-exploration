import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State
from dash import Input, Output, State, ALL, callback, ctx


# initialize app
dash.register_page(__name__, path="/exploring_intent")

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
            ),),
        dbc.Row(dcc.Graph(id="intent_exploring_graph", style={'width': '80%', 'height': '100vh'}), justify="center"),
        dbc.Row([html.H1("Intent Exploration", style={'textAlign': 'center'}),
                    html.Br(),
                    html.Br(),
                    html.Div(children=[], id="results_exploring_playlists",
                             style={'padding': 32,
                                    "justifyContent": "center",
                                    'flex-direction': 'row',
                                    'display': 'block'})])
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

    #fig.update_layout(
    #    autosize=False,
    #    width=1600,
    #    height=800)

    return fig


@callback(
    [Output("intent_exploring_graph", "figure"),
     Output("results_exploring_playlists", "children")],
    [
        Input("choice_model", "value"),
    ]
)
def on_playlist_name_changed(model_name):
    df_test_data = pd.DataFrame({"x": [10, 1, 1, 0], "y": [1, 1, 0, -10], "model": ["cross", "cross", "ensemble", "ensemble"], 'playlist_name': ["a", "b", "a", "b"], 'intent': [0, 1, 0, 1]})

    fig = make_fig(df_test_data, model_name)
    children = [model_name]
    return fig, children
