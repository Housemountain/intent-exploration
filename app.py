import dash
import dash_bootstrap_components as dbc

from dash import html

# initialize app
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.UNITED],
                use_pages=True,
                meta_tags=[{
                    'name': 'viewport',
                    'content': 'width=device-width, initial-scale=1.0'
                }])

# set app layout
app.layout = html.Div(children=[
    dash.page_container
])


if __name__ == "__main__":
    print("Start.")
    app.run_server(debug=True)
