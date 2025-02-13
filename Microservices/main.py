import random
from dash import Dash, html, dash_table, dcc
import pandas as pd
import plotly.express as px

import pandapower as pp
import pandapower.networks as pn
import matplotlib.pyplot as plt
import dash
import dash_core_components as dcc
import dash_html_components as html
import base64
import pandas as pd
from io import BytesIO
from dash.dependencies import Input, Output ,State

max_size_nodes = 24
nodes = 18
lines = nodes-1
loads = 9
num_intervals = 48
num_scenarios = 5
time_steps_simulation = num_intervals * 14
base_output_path = f"./Data"
common_folder=f"{base_output_path}/train_data/AV3/case_net_generate_n_15_scenario_0/load/load.csv"

# Incorporate data
df = pd.read_csv(common_folder ,header=[0, 1, 2])
df.columns = ['_'.join(map(str, col)).strip().split('_Unnamed')[0] for col in df.columns.values]
df = df.rename(columns={"Unnamed: 0_level_0_node_id_time":"step_time"})
# App layout
res_load_p_columns = [column for column in df.columns if 'res_load_p' in column]
res_load_q_columns = [column for column in df.columns if 'res_load_q' in column]
vm_columns = [column for column in df.columns if 'vm' in column]



# Generar una imagen del gráfico de la red
# fig, ax = plt.subplots()
# pp.plotting.simple_plot(net, ax=ax)

# Convertir la figura en una imagen en formato base64
# buffer = BytesIO()
# fig.savefig(buffer, format='png')
# buffer.seek(0)
# image_base64 = base64.b64encode(buffer.read()).decode('utf-8')

# Función para generar la imagen de la red


def create_network(net_type=0):
    if net_type == 0:
        net =pn.simple_four_bus_system()
    elif net_type == 1:   
        net =pn.case24_ieee_rts()
    elif net_type == 2:
        net = pn.case30()
    else:
        net = pn.case14()
    return net


def generate_image():
    # Crear una red de ejemplo con pandapower
    net = create_network(random.randint(0,3))

    # Crear una barra de referencia externa (ext_grid) en la red
    slack_bus = 0
    pp.create_ext_grid(net, bus=slack_bus, vm_pu=1.02)

    fig, ax = plt.subplots()
    pp.plotting.simple_plot(net, ax=ax)
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return image_base64



# Generar la imagen inicial
initial_image = generate_image()




external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css']

# Crear la aplicación Dash
app = dash.Dash(__name__) # , external_stylesheets=external_stylesheets

# # Layout de la aplicación
# app.layout = html.Div(style={'padding': '20px'}, children=[
#     html.H1(children='Electric Network Reconstruction System ENRS'),

#     html.Button('Actualizar Imagen', id='update-button', n_clicks=1),
#     html.Img(id='graph-image', style={'padding': '20px'}),

#     html.Div(className='row', children=[
#         html.Div(className='col-md-6', children=[
#             html.Button('Actualizar Imagen', id='update-button', n_clicks=0, className='btn btn-primary mb-3'),
#             html.Img(id='graph-image', className='img-fluid', style={'padding': '20px'}),
#         ]),
#         html.Div(className='col-md-6', children=[
#             html.Div(className='form-group', children=[
#                 html.Label('Numero de Nodos:', className='form-label'),
#                 dcc.Input(id='input-integer-1', type='number', value=0, className='form-control'),
#             ]),
#             html.Div(className='form-group', children=[
#                 html.Label('Valor Entero 2:', className='form-label'),
#                 dcc.Input(id='input-integer-2', type='number', value=0, className='form-control'),
#             ]),
#             html.Div(className='form-group', children=[
#                 html.Label('Texto 1:', className='form-label'),
#                 dcc.Input(id='input-text-1', type='text', value='', className='form-control'),
#             ]),
#             html.Div(className='form-group', children=[
#                 html.Label('Texto 2:', className='form-label'),
#                 dcc.Input(id='input-text-2', type='text', value='', className='form-control'),
#             ]),
#             html.Div(className='form-group', children=[
#                 html.Label('Texto 3:', className='form-label'),
#                 dcc.Input(id='input-text-3', type='text', value='', className='form-control'),
#             ]),
#             html.Button('Actualizar Formulario', id='update-form-button', n_clicks=0, className='btn btn-success mt-3')
#         ]),
#     ]),
#     html.Div(id='output-form', className='mt-4', style={'padding': '20px'}),

#     dcc.Dropdown(
#         id='histogram-selector',
#         options=[
#             {'label': 'Histogram Cargas Activas', 'value': 'res_load_p'},
#             {'label': 'Histogram Cargas Reactivas', 'value': 'res_load_q'}
#         ],
#         value='res_load_p'  # Valor por defecto
#     ),
    
#     dcc.Graph(id='histogram-graph', style={'padding': '20px'}),

#     dcc.Dropdown(
#         id='table-selector',
#         options=[
#             {'label': 'Data res_load_p', 'value': 'res_load_p'},
#             {'label': 'Data res_load_q', 'value': 'res_load_q'},
#             {'label': 'Data vm_pu', 'value': 'vm_pu'}
#         ],
#         value='res_load_p'  # Valor por defecto
#     ),

#     # Tabla con scroll
#     html.Div(style={'overflow': 'auto', 'height': '400px'}, children=[
#         dash_table.DataTable(
#             id='data-table',
#             columns=[{'name': i, 'id': i} for i in df.columns],
#             page_size=10,
#             style_table={'height': '100%', 'overflowX': 'auto'},
#             style_cell={'padding': '10px'}
#         ),
#     ]),
# ])


#  Layout de la aplicación
app.layout = html.Div(className='container', children=[
    html.H1('Electric Network Reconstruction System ENRS', className='text-center my-4'),
    html.Div(className='row', children=[
        html.Div(className='col-md-6', children=[
            html.Button('Actualizar Imagen', id='update-button', n_clicks=0, className='btn btn-primary mb-3'),
            html.Img(id='graph-image', className='img-fluid', style={'padding': '20px'}),
        ]),
        html.Div(className='col-md-6', children=[
            html.Img(id='graph-image', className='img-fluid', style={'padding': '20px'}),
            html.Img(id='graph-image', className='img-fluid', style={'padding': '20px'}),
        ]),
        html.Div(className='col-md-6', children=[
            html.Div(className='form-group', children=[
                html.Label('Numero de Nodos:', className='form-label'),
                dcc.Input(id='input-integer-1', type='number', value=0, className='form-control'),
            ]),
            html.Div(className='form-group', children=[
                html.Label('Valor Entero 2:', className='form-label'),
                dcc.Input(id='input-integer-2', type='number', value=0, className='form-control'),
            ]),
            html.Div(className='form-group', children=[
                html.Label('Texto 1:', className='form-label'),
                dcc.Input(id='input-text-1', type='text', value='', className='form-control'),
            ]),
            html.Div(className='form-group', children=[
                html.Label('Texto 2:', className='form-label'),
                dcc.Input(id='input-text-2', type='text', value='', className='form-control'),
            ]),
            html.Div(className='form-group', children=[
                html.Label('Texto 3:', className='form-label'),
                dcc.Input(id='input-text-3', type='text', value='', className='form-control'),
            ]),
            html.Button('Actualizar Formulario', id='update-form-button', n_clicks=0, className='btn btn-success mt-3'),
        ]),
    ]),
    html.Div(id='output-form', className='mt-4', style={'padding': '20px'}),
    dcc.Dropdown(
        id='histogram-selector',
        options=[
            {'label': 'Histogram Cargas Activas', 'value': 'res_load_p'},
            {'label': 'Histogram Cargas Reactivas', 'value': 'res_load_q'}
        ],
        value='res_load_p'  # Valor por defecto
    ),
    dcc.Graph(id='histogram-graph', style={'padding': '20px'}),
    dcc.Dropdown(
        id='table-selector',
        options=[
            {'label': 'Data res_load_p', 'value': 'res_load_p'},
            {'label': 'Data res_load_q', 'value': 'res_load_q'},
            {'label': 'Data vm_pu', 'value': 'vm_pu'}
        ],
        value='res_load_p'  # Valor por defecto
    ),
    html.Div(style={'overflow': 'auto', 'height': '400px'}, children=[
        dash_table.DataTable(
            id='data-table',
            columns=[{'name': i, 'id': i} for i in df.columns],
            page_size=10,
            style_table={'height': '100%', 'overflowX': 'auto'},
            style_cell={'padding': '10px'}
        ),
    ]),
])

@app.callback(
    Output('output-form', 'children'),
    [Input('update-form-button', 'n_clicks')],
    [State('input-integer-1', 'value'),
     State('input-integer-2', 'value'),
     State('input-text-1', 'value'),
     State('input-text-2', 'value'),
     State('input-text-3', 'value')]
)
def update_form(n_clicks, int1, int2, text1, text2, text3):
    return html.Div([
        html.P(f'Valor Entero 1dftgg: {int1}'),
        html.P(f'Valor Entero 2: {int2}'),
        html.P(f'Texto 1: {text1}'),
        html.P(f'Texto 2: {text2}'),
        html.P(f'Texto 3: {text3}')
    ])



@app.callback(
    Output('graph-image', 'src'),
    [Input('update-button', 'n_clicks')]
)
def update_image(n_clicks):
    image_base64 = generate_image()

    return 'data:image/png;base64,{}'.format(image_base64)


@app.callback(
    Output('data-table', 'data'),
    [Input('table-selector', 'value')]
)
def update_table(selected_column):
    table_data = df.to_dict('records')
    return table_data

@app.callback(
    Output('histogram-graph', 'figure'),
    [Input('histogram-selector', 'value')]
)
def update_histogram(selected_data):
    selected_column = {
            "res_load_p" : res_load_p_columns ,
            "res_load_q": res_load_q_columns,
            "vm_pu": vm_columns, 
        }
        
    fig = px.histogram(df, x='step_time', y=selected_column[selected_data], histfunc='avg')
    return fig

if __name__ == '__main__':
    app.run(debug=True)
