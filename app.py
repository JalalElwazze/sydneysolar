import dash
import dash_html_components as html
from elements import Elements
from dash.dependencies import Input, Output, State
from data import Data
import plotly.graph_objs as go
from predictor import Predictor

app = dash.Dash(__name__)
server = app.server
app.title = "Sydney Solar | Exposure predictor"
external_css = ["https://fonts.googleapis.com/css?family=Product+Sans:400,400i,700,700i",
                "https://rawgit.com/JalalElwazze/sydneysolar/master/app.css"]

for css in external_css:
    app.css.append_css({"external_url": css})

app.layout = html.Div(
    [
        Elements.header(),
        Elements.graph_block(),
        Elements.scatter_spear_block(),
        Elements.prediction_block(),
        Elements.convergence_block()
    ])

dataset = Data.merged_norm()

# Dropdown --> Time series
@app.callback(
    Output("raw_data_graph", "figure"), [Input("raw_data_select", "value")]
)
def update_raw_data_plot(values):

    new_data = []
    colors = ['#4f9982', '#ba554f', "#f49b41", '#458fd3']

    for index, value in enumerate(values):
        new_data.append(
            dict(
                x=dataset['Time'],
                y=dataset[value],
                name=value,
                type='line',
                line=dict(color=colors[index], width=2),
            )
        )

        layout = go.Layout(
            title="Raw Time Series Data",
            margin={'l': 50, 'b': 50, 't': 70, 'r': 0},
        )

    return {'data': new_data, 'layout': layout}


# Dropdon --> Scatter
@app.callback(
    Output("scatter_plot", "figure"), [Input("scatter_select", "value")]
)
def update_scatter_plot(values):
    new_data = []
    colors = ['#4f9982', '#ba554f', "#f49b41", '#458fd3']

    for index, value in enumerate(values):
        new_data.append(
            go.Scatter(
                x=dataset['Exposure'],
                y=dataset[value],
                name=value,
                mode='markers', opacity=0.7,
                marker={
                    'size': 15,
                    'line': {'width': 0.5, 'color': 'white'},
                    'color': colors[index]
                },
            )
        )

    layout = go.Layout(
        title="Scatter Vs Solar Exposure",
        margin={'l': 50, 'b': 50, 't': 100, 'r': 50},
    )

    return {'data': new_data, 'layout': layout}


# Dropdown --> Pie
@app.callback(
    Output("spearman_pie", "figure"), [Input("scatter_select", "value")]
)
def update_pie(values):
    new_values = []
    new_colors = []
    colors = ['#4f9982', '#ba554f', "#f49b41", '#458fd3']

    for index, value in enumerate(values):
        new_values.append(Data.compute_rank(value))
        new_colors.append(colors[index])

    new_data = go.Pie(
        labels=values, values=new_values, hoverinfo='label+value', textinfo='percent', textfont=dict(size=15),
        showlegend=False, opacity=0.9,
        marker=dict(colors=new_colors, line=dict(color='white', width=2)))

    layout = go.Layout(
        title="Spearman Ranks",
        margin={'l': 50, 'b': 50, 't': 100, 'r': 50},
    )

    return {'data': [new_data], 'layout': layout}


# Prediction tools --> Train Model
@app.callback(
    Output("prediction_graph", "style"), [Input('train', 'n_clicks')],
    [State("select_inputs", "value"), State("select_model", "value")]
)
def train_model(static, inputs, models):
    if (inputs is not None) and (models is not None):
        # Filter inputs
        file = models
        inputs = sorted(inputs)

        # Make filename
        for parameter in inputs:
            temp = parameter.strip(" ")
            file += "_" + temp

        # Initialise model
        model = Predictor(dataset, inputs=inputs, model=models, filename=file)

        # Train Model
        model.train(100)


# Prediction tools --> Visualise on Main Plot
@app.callback(
    Output("prediction_graph", "figure"), [Input('run', 'n_clicks')],
    [State("select_inputs", "value"), State("select_model", "value")]
)
def run_model(static, inputs, models):

    if (inputs is not None) and (models is not None):

        # Filter inputs
        file = models
        inputs = sorted(inputs)

        # Make filename
        for parameter in inputs:
            temp = parameter.strip(" ")
            file += "_" + temp

        # Initialise model
        model = Predictor(dataset, inputs=inputs, model=models, filename=file)

        # Run Model
        result = model.predict()

        # Visualise
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

        new_data = [{'x': dataset.Time, 'y': dataset["Exposure"], 'name': "True Value",
                     'line': dict(color='rgba(68, 68, 68, 0.4)', width=2)}, lower_bound, trace, upper_bound]

        return {'data': new_data}


# Prediction Tools --> Status Update color
@app.callback(
    Output("status", "style"), [Input('train', 'n_clicks'), Input('run', 'n_clicks')],
)
def button_color(static, clicks_run):
    if clicks_run is not None:
        if (clicks_run % 2) == 0:
            color = 'rgba(79, 153, 130, 0.7)'
        else:
            color = 'rgba(244, 155, 65, 0.7)'

        return {'width': '150px', 'margin-right': '10px', 'margin-top': '10px',
                'background-color': color}


# Run Button --> Fit Convergence
@app.callback(
    Output("fit_convergence_plot", "figure"), [Input('run', 'n_clicks')],
    [State("select_inputs", "value"), State("select_model", "value")],
)
def update_converge(static, inputs, models):

    if (inputs is not None) and (models is not None):
        # Filter inputs
        file = models
        inputs = sorted(inputs)

        # Make filename
        for parameter in inputs:
            temp = parameter.strip(" ")
            file += "_" + temp

        # Initialise model
        model = Predictor(dataset, inputs=inputs, model=models, filename=file)

        # Run
        results = model.check()

        return {'data': [{'x': results[-1], 'y': results[0], 'line': dict(color='grey')}]}


# Run Button --> Prediction Convergence
@app.callback(
    Output("pred_convergence_plot", "figure"), [Input('run', 'n_clicks')],
    [State("select_inputs", "value"), State("select_model", "value")],
)
def update_prediction_convergence(static, inputs, models):

    if (inputs is not None) and (models is not None):
        # Filter inputs
        file = models
        inputs = sorted(inputs)

        # Make filename
        for parameter in inputs:
            temp = parameter.strip(" ")
            file += "_" + temp

        # Initialise model
        model = Predictor(dataset, inputs=inputs, model=models, filename=file)

        # Run
        results = model.check()

        return {'data': [{'x': results[-1], 'y': results[1], 'line': dict(color='black')}]}

if __name__ == '__main__':
    app.run_server(debug=True)
