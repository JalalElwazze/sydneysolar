import dash_html_components as html
import dash_core_components as dcc
from data import Data
import plotly.graph_objs as go
from predictor import Predictor

# Global
dataset = Data.merged_norm()


class Elements:

    @staticmethod
    def header():

        layout = html.Div(
            [
                html.H2('Exposure Predictor',
                        style={'display': 'inline',
                               'float': 'left',
                               'font-size': '2.65em',
                               'margin-left': '7px',
                               'font-weight': 'bolder',
                               'font-family': 'Product Sans',
                               'color': "rgba(117, 117, 117, 0.95)",
                               'margin-top': '20px',
                               'margin-bottom': '0'
                               }),
                html.Img(src="https://s13.postimg.org/ulq0o1wlz/Sydney_Solar.png",
                        style={
                            'height': '50px',
                            'float': 'right',
                            'margin-top': '25px',
                        },
                ),
            ], className='container', style={'height': '100px', 'margin-top': '10px'}
        )

        return layout

    @staticmethod
    def graph_block():
        layout = html.Div(
            [
                # Dropdown Menu
                html.Div(
                    [
                        dcc.Dropdown(
                            id='raw_data_select',
                            options=[{'label': i, 'value': i} for i in dataset.columns[1:]],
                            multi=True,
                            value=["Exposure", "Max Temp", 'Rain', 'Min Temp']
                        ),
                    ],
                ),

                html.Div(
                    # Graph
                    dcc.Graph(
                        id='raw_data_graph', config={'displayModeBar': False}, animate=True,
                        figure=
                        {
                            'data': [{'x': dataset['Time'], 'y': dataset["Exposure"],
                                      'name': "Exposure", 'line': dict(color='#4f9982', width=2)},
                                     {'x': dataset['Time'], 'y': dataset["Max Temp"],
                                      'name': "Max Temp", 'line': dict(color='#ba554f', width=2)},
                                     {'x': dataset['Time'], 'y': dataset["Rain"],
                                      'name': "Rain", 'line': dict(color='#f49b41', width=2)},
                                     {'x': dataset['Time'], 'y': dataset["Min Temp"],
                                      'name': "Min Temp", 'line': dict(color='#458fd3', width=2)}
                                     ],

                            'layout': go.Layout(
                                title="Raw Time Series Data",
                                margin={'l': 50, 'b': 50, 't': 70, 'r': 0},
                            )
                        }
                    ), style={'margin-top': '10px'}
                )

            ], className='container', style={'margin-top': '10px'}
        )

        return layout

    @staticmethod
    def scatter_spear_block():
        layout = html.Div(
            [

                # Dropdown
                html.Div(
                    dcc.Dropdown(
                        id='scatter_select',
                        multi=True,
                        options=[{'label': i, 'value': i} for i in dataset.columns[1:]],
                        value=['Min Temp', 'Max Temp', 'Rain']
                    )
                ),

                # Scatter
                html.Div(
                    dcc.Graph(
                        id='scatter_plot', animate=False, config={'displayModeBar': False},
                        figure=
                        {
                            'data': [
                                go.Scatter(
                                    x=dataset['Exposure'],
                                    y=dataset['Min Temp'],
                                    name='Min Temp',
                                    mode='markers', opacity=0.7,
                                    marker={
                                        'size': 15,
                                        'line': {'width': 0.5, 'color': 'white'},
                                        'color': "#4f9982"
                                    },
                                ),

                                go.Scatter(
                                    x=dataset['Exposure'],
                                    y=dataset['Max Temp'],
                                    name='Max Temp',
                                    mode='markers', opacity=0.7,
                                    marker={
                                        'size': 15,
                                        'line': {'width': 0.5, 'color': 'white'},
                                        'color': "#ba554f"
                                    },
                                ),

                                go.Scatter(
                                    x=dataset['Exposure'],
                                    y=dataset['Rain'],
                                    name='Rain',
                                    mode='markers', opacity=0.7,
                                    marker={
                                        'size': 15,
                                        'line': {'width': 0.5, 'color': 'white'},
                                        'color': "#f49b41"
                                    },
                                ),
                            ],
                            'layout': go.Layout(
                                title="Scatter Vs Solar Exposure",
                                margin={'l': 50, 'b': 50, 't': 100, 'r': 50},
                            )
                        },
                    ), style={'width': '67%', 'display': 'inline-block'}
                ),

                # Pie Chart
                html.Div(
                    [
                        html.Div(
                            dcc.Graph(
                                id='spearman_pie', config={'displayModeBar': False},
                                figure=
                                {
                                    'data': [
                                        go.Pie(labels=['Min Temp', 'Max Temp', 'Rain'],
                                               values=[Data.compute_rank('Min Temp'),
                                                       Data.compute_rank('Max Temp'),
                                                       Data.compute_rank('Rain')],
                                               hoverinfo='label+value', textinfo='percent', textfont=dict(size=15),
                                               showlegend=False, opacity=0.9,
                                               marker=dict(colors=['#4f9982', '#ba554f', "#f49b41", '#458fd3'],
                                                           line=dict(color='white', width=2)))
                                    ],
                                    'layout': go.Layout(
                                        title="Spearman Ranks",
                                        margin={'l': 50, 'b': 50, 't': 100, 'r': 50},
                                    )

                                },
                            ),
                        )
                    ], style={'width': '33%', 'display': 'inline-block'}
                )

            ], className='container', style={'margin-top': '10px'},
        )

        return layout

    @staticmethod
    def prediction_block():
        model = Predictor(dataset, inputs=['Max Temp', 'Min Temp', 'Rain'], model='svr',
                          filename='svr_Max Temp_Min Temp_Rain')
        result = model.predict()
        error = 0.10
        upper_bound = go.Scatter(
            name='Upper Error', x=result['Time'], y=result['Predictions']*(1 + error),
            mode='lines', marker=dict(color="444"), line=dict(width=0), fillcolor='rgba(62, 120, 214, 0.3)',
            fill='tonexty')

        trace = go.Scatter(
            name='Predicted', x=result['Time'], y=result['Predictions'],
            mode='lines', line=dict(color='rgba(62, 120, 214, 0.7)'),
            fillcolor='rgba(62, 120, 214, 0.3)', fill='tonexty',)

        lower_bound = go.Scatter(
            name='Lower Error', x=result['Time'], y=result['Predictions']*(1 - error),
            marker=dict(color="444"), line=dict(width=0), mode='lines')

        layout = html.Div(
            [
                # Buttons and Controls
                html.Div(
                    [
                        html.Div(
                            dcc.Dropdown(
                                options=[
                                    {'label': 'K Nearest Neighbours', 'value': 'nnr'},
                                    {'label': 'Multiple Linear Regression', 'value': 'lmr'},
                                    {'label': 'Support Vector Regression', 'value': 'svr'},
                                    {'label': 'Gradient Boosted Regression', 'value': 'gbr'},
                                    {'label': 'Kernel Ridge Regression', 'value': 'krr'},
                                    {'label': 'ADA Boost Regression', 'value': 'ada'},
                                    {'label': 'Bayesian Ridge Regression', 'value': 'brr'},
                                    {'label': 'Gaussian Process Regression', 'value': 'gpr'},
                                    {'label': 'Neural Network Regression', 'value': 'mlp'},
                                ],
                                placeholder='Select a Model', id='select_model',
                            ), style={'width': '47.5%', 'display': 'inline-block', 'margin-right': '10px'}
                        ),

                        html.Div(
                            dcc.Dropdown(
                                options=[
                                    {'label': 'Maximum Temperature', 'value': 'Max Temp'},
                                    {'label': 'Minimum Temperature', 'value': 'Min Temp'},
                                    {'label': 'Rainfall', 'value': 'Rain'},
                                ],
                                multi=True, placeholder='Select Inputs', id='select_inputs'
                            ), style={'width': '47.5%', 'display': 'inline-block'}

                        ),
                    ]
                ),

                html.Div(
                    # Graph
                    dcc.Graph(
                        id='prediction_graph', config={'displayModeBar': False}, animate=True,
                        figure=
                        {
                            'data': [{'x': dataset.Time, 'y': dataset["Exposure"],
                                      'name': "True Value", 'line': dict(color='rgba(68, 68, 68, 0.4)', width=2)},
                                     lower_bound, trace, upper_bound
                                     ],

                            'layout': go.Layout(
                                title="Exposure Prediction",
                                margin={'l': 50, 'b': 50, 't': 70, 'r': 50},
                                showlegend=False,
                                xaxis={'range': [dataset.Time.iloc[-100], dataset.Time.iloc[-1]]}
                            )
                        }
                    ), style={'margin-top': '10px'}
                ),

                html.Button(
                    "Train", id='train', className='button1',
                    style={'width': '150px', 'margin-right': '10px', 'margin-top': '10px'}
                ),

                html.Button(
                    "Ready", className='button1', id='status',
                    style={'width': '150px', 'margin-right': '10px', 'margin-top': '10px',
                           'background-color': 'rgba(79, 153, 130, 0.7)'}
                ),
                html.Button(
                    "Run", id='run', className='button1', style={'width': '150px', 'margin-top': '10px'}
                )

            ], className='container', style={'margin-top': '10px', 'textAlign': 'center'}
        )

        return layout

    @staticmethod
    def convergence_block():
        model = Predictor(dataset, inputs=['Max Temp', 'Min Temp', 'Rain'], model='svr',
                          filename='svr_Max Temp_Min Temp_Rain')
        results = model.check()

        layout = html.Div(
            [
                # Left Plot
                html.Div(
                    dcc.Graph(
                        id='fit_convergence_plot', config={'displayModeBar': False}, animate=True,
                        figure=
                        {
                            'data': [{'x': results[-1], 'y': results[0], 'line': dict(color='grey')}],
                            'layout': go.Layout(
                                title="Loss Scores: Fitting",
                                xaxis={'title': "Number of Data Points"},
                                yaxis={'range': [0, 0.3]},
                                margin={'l': 50, 'b': 50, 't': 100, 'r': 50},
                            )
                        },
                    ), style={'width': '50%', 'display': 'inline-block'}
                ),

                # Right Plot
                html.Div(
                    dcc.Graph(
                        id='pred_convergence_plot', config={'displayModeBar': False}, animate=True,
                        figure=
                        {
                            'data': [{'x': results[-1], 'y': results[1], 'line': dict(color='black')}],
                            'layout': go.Layout(
                                title="Loss Scores: Prediction",
                                xaxis={'title': "Number of Data Points"},
                                yaxis={'range': [0, 0.3]},
                                margin={'l': 50, 'b': 50, 't': 100, 'r': 50},
                            )
                        },
                    ), style={'width': '50%', 'display': 'inline-block'}
                ),
            ], className='container', style={'margin-top': '10px'},
        )

        return layout


