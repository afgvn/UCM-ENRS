import os
import random
import base64
import joblib
import numpy as np
import pandas as pd
from io import BytesIO ,StringIO

import matplotlib
matplotlib.use('Agg')

import plotly.express as px
import pandapower as pp
import pandapower.networks as pn
import matplotlib.pyplot as mplt

from dash import Dash, html, dash_table, dcc
from dash.dependencies import Input, Output ,State
from sklearn.preprocessing import  MinMaxScaler 

max_size_nodes = 24
nodes = 18
lines = nodes-1
loads = 9
num_intervals = 48
num_scenarios = 5
time_steps_simulation = num_intervals * 14

model_output_path ="./Microservices/Modelos/{}"
model_default = joblib.load(model_output_path.format("model_lstm_cnn_V3_final"))
model_files = [os.path.join(model_output_path.format(""), str(nombre)) for nombre in os.listdir(model_output_path.format(""))]
model_names_files = [nombre for nombre in os.listdir(model_output_path.format(""))]
static_image_path = '/assets/logo_ucm_ntic.png'

base_output_path = f"./Microservices/Data"
common_folder=f"{base_output_path}/default"
loads_files=f"{common_folder}/load/load.csv"
matrix_file=f"{common_folder}/indicen_matriz/incidence_matrix.csv" 
df_matrix = pd.read_csv(matrix_file ,header=[0])
df_matrix.rename(columns={"Unnamed: 0":"lines_id"},inplace=True)
df_matrix.set_index("lines_id",inplace=True)
df_matrix_gen = pd.read_csv(matrix_file ,header=[0])
df_matrix_gen.rename(columns={"Unnamed: 0":"lines_id"},inplace=True)
df_matrix_gen.set_index("lines_id",inplace=True)
df_loads = pd.read_csv(loads_files ,header=[0])

select_columns_table = df_loads.columns
res_load_p_columns = [column for column in df_loads.columns if 'res_p' in column]
res_load_q_columns = [column for column in df_loads.columns if 'res_q' in column]
vm_columns = [column for column in df_loads.columns if 'vm' in column]
va_degree_columns = [column for column in df_loads.columns if 'va_degree' in column]
external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css']


@staticmethod
def ajustar_celdas(df:pd.DataFrame,y=0.4):
    df_nuevo = pd.DataFrame(0, index=df.index, columns=df.columns)
    for col in df.columns:
        top_values = df[col].nlargest(2)
        if len(top_values) > 0:
            top_indices = top_values.index
            # if top_values.sum() < y :
            #     continue
            if len(top_indices) > 0:
                df_nuevo.loc[top_indices[0], col] =3
            if len(top_indices) > 1:
                df_nuevo.loc[top_indices[1], col] = 5
    return df_nuevo

@staticmethod
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

@staticmethod
def print_bw_matrix(df:pd.DataFrame):
    fig, ax = mplt.subplots(figsize=(5, 5))
    mplt.matshow(df, cmap='gray', fignum=fig.number)
    mplt.xticks(ticks=np.arange(df.shape[1]), labels=df.columns)
    mplt.yticks(ticks=np.arange(df.shape[0]), labels=df.index)
    mplt.colorbar()
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    mplt.close(fig)
    return image_base64

@staticmethod
def plot_simple_df_net(df:pd.DataFrame):
    df = df.transpose()
    net = pp.create_empty_network()
    buses = [pp.create_bus(net, vn_kv=110, name=f"Bus {bus}") for bus in range(len(df.index))]
    number_line = 0

    for col in df.columns:
        from_bus = None
        to_bus = None
        for i, value in df[col].items():
                if value == 5 and from_bus is None:
                        from_bus = int(i)
                if value == 3 and to_bus is None:
                        to_bus = int(i)
        
        if  (not from_bus is None and 
        not to_bus is None ): 
                length_km = 10
                pp.create_line(net, name=f"number_line{number_line}", from_bus=from_bus, to_bus=to_bus, length_km=length_km, std_type="NAYY 4x50 SE")
                number_line += 1

    slack_bus = 0 
    pp.create_gen(net, bus=slack_bus, p_mw=random.uniform(5.0, 10.0), vm_pu=1.02, slack=True, name="Slack Gen")
    other_gen_buses = random.sample([b for b in buses if b != slack_bus], k=max(1,  len(df.index) // 4))
    for bus in other_gen_buses:
        pp.create_gen(net, bus=bus, p_mw=random.uniform(3.0, 6.0), vm_pu=1.02, name=f"Gen {bus}")

    pp.create_ext_grid(net, bus=slack_bus, vm_pu=1.02)
    fig, ax = mplt.subplots(figsize=(5, 5))
    pp.plotting.simple_plot(net, ax=ax)
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return image_base64

@staticmethod
def generate_image():
    net = create_network(random.randint(0,3))
    slack_bus = 0
    pp.create_ext_grid(net, bus=slack_bus, vm_pu=1.02)
    fig, ax = mplt.subplots(figsize=(5, 5))
    pp.plotting.simple_plot(net, ax=ax)
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return image_base64

@staticmethod
def add_columns(df, prefix, total_columns):
   columns = [col.split("_")[-1] for col in df.columns if prefix in col ] 
   max_col = max(columns)
   for col in range( int(max_col) , total_columns+1):
       column = f" {prefix}{col}"
       df[column]= 0.0

   return df

@staticmethod
def execute_model():
    scaler = MinMaxScaler()
    df_loads_temp = df_loads.set_index("time_step")
    df_loads_temp = add_columns(df_loads_temp, 'va_degree_', max_size_nodes)
    df_loads_temp = add_columns(df_loads_temp, 'res_p_', max_size_nodes)
    df_loads_temp = add_columns(df_loads_temp, 'res_q_', max_size_nodes)
    df_loads_temp = add_columns(df_loads_temp, 'vm_pu_', max_size_nodes)
    df_loads_temp = df_loads_temp.fillna(0)
    columnas_excluir = ['nodes_numbers', 'loads_number']
    df_loads_temp[columnas_excluir] = df_loads_temp[columnas_excluir]
    df_restantes = df_loads_temp.drop(columns=columnas_excluir).astype(float)
    df_loads_temp = pd.concat([df_loads_temp[columnas_excluir], df_restantes], axis=1)
    df_loads_temp = df_loads_temp.sort_index(axis=1)

    X = df_loads_temp.values
    X =  scaler.fit_transform(X)
    num_samples = len(X) // num_intervals
    X = X.reshape((num_samples,num_intervals,len(df_loads_temp.columns)))
    try:
        predicted_image = model_default.predict(X)
        test_image = predicted_image[0][0]
        data_out = []
        for i in range(test_image.shape[0]):
            row = []
            for j in range(test_image.shape[1]):
                val = test_image[i, j]
                row.append(val)
            data_out.append(row)
        
        df_topo_out = pd.DataFrame(data_out, columns=[f'{i}' for i in range(test_image.shape[1])])
        df_topo_out = ajustar_celdas(df_topo_out,y=0.0)
        df_topo_out =df_topo_out.transpose()
        image_bw = print_bw_matrix(df_topo_out)
        image_net = plot_simple_df_net(df_topo_out)
        print("Update image")
    except Exception as error:
        print(f"Fail Update image :  {error}")
        image_bw = print_bw_matrix(df_matrix_gen)
        image_net = plot_simple_df_net(df_matrix_gen)
    return image_bw ,image_net


app = Dash(__name__ ,external_stylesheets=external_stylesheets) # 
server = app.server 
app.config.prevent_initial_callbacks = 'initial_duplicate'


app.layout = html.Div(className='container', children=[
    html.H1('Electric Network Reconstruction System ENRS', className='text-center my-4'),
    html.Div(className='row', children=[
        html.Div(className='col-md-6', children=[
            html.Div(className='form-group', children=[
                    html.Label(' Elija tipo de archivo para realizar accion ', className='form-label'),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Cargas o Matriz de incidencia'
                        ]),
                        style={
                            'width': '50%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '15px'
                        },
                        multiple=False
                    ),
                ]),
            html.Div(id='dynamic-content', className='col-md-8'),
            dcc.Dropdown(
                            id='model-selector',
                            options=[{'label': model, 'value': model} for model in model_names_files],
                            value=model_names_files[0],
                            style={'display': 'none'}
                        ),
            html.Button('Ejecutar', id='execute-button', n_clicks=0, className='btn btn-success mt-3', style={'display': 'none'}),
            html.Button('Actualizar Imagen', id='graph-button', n_clicks=0, className='btn btn-primary mt-3', style={'display': 'none'}),
        ]),
        html.Div(className='col-md-4', children=[
            html.Img(id="static-img", src=static_image_path, className='img-fluid', style={'padding': '30px'}),
        ]),
    ]),
    html.H3('Red Original', className='text-center my-4'),
    html.Div(className='row', children=[
        html.Div(className='col-md-6', children=[
            html.Img(id='graph-image', className='img-fluid', style={'padding': '30px'})
        ]),
        html.Div(className='col-md-6', children=[
            html.Img(id='graph-image-2', className='img-fluid', style={'padding': '30px'}),
        ]),
    ]),
    html.H3('Red Generada', className='text-center my-4'),
    html.Div(className='row', children=[
        html.Div(className='col-md-6', children=[
            html.Img(id='graph-matrix-gen', className='img-fluid', style={'padding': '30px'})
        ]),
        html.Div(className='col-md-6', children=[
            html.Img(id='graph-image-gen', className='img-fluid', style={'padding': '30px'}),
        ]),
    ]),
    html.Div(className='row', children=[
        html.Div(className='col-md-6', children=[
                html.Label('Seleccionar Variable de Nodo:', className='form-label'),
                dcc.Dropdown(
                    id='histogram-selector',
                    options=[
                        {'label': 'Histogram Cargas Activas', 'value': 'res_p'},
                        {'label': 'Histogram Cargas Reactivas', 'value': 'res_q'},
                        {'label': 'Histogram Valor por unidad de media', 'value': 'vm_pu'},
                        {'label': 'Histogram angulo del voltage', 'value': 'va_degree'},
                    ],
                    value='res_p',
                ),
        ]),
    ]),
    html.Div(className='row', children=[
        html.Div(className='col-md-6', children=[
                dcc.Graph(id='histogram-graph', style={'padding': '20px'}),
        ]),
        html.Div(className='col-md-6', children=[
            html.Div(className='table-responsive', children=[
              dash_table.DataTable(
                    id='data-table',
                    page_size=10,
                    columns=[{'name': i, 'id': i} for i in df_loads.columns],       
                    style_table={'height': '100%', 'overflowX': 'auto'},
                    style_cell={
                        'padding': '10px',
                        'textAlign': 'left',
                        'whiteSpace': 'normal',
                        'height': 'auto',
                        'border': '1px solid grey'
                    },
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold',
                        'border': '1px solid black'
                    },
                    style_data={
                        'border': '1px solid grey'
                    }
                ),
            ]),
        ]),
    ]),
])

@app.callback(
    [Output('graph-image', 'src'),
    Output('graph-image-2', 'src')],
    [Input('graph-button', 'n_clicks')]
)
def update_image(n_clicks):
    image_bw = print_bw_matrix(df_matrix)
    image_net = plot_simple_df_net(df_matrix)
    return ['data:image/png;base64,{}'.format(image_bw),
    'data:image/png;base64,{}'.format(image_net)]

@app.callback(
    [Output('graph-matrix-gen', 'src'),
    Output('graph-image-gen', 'src')],
    [Input('execute-button', 'n_clicks')]
)
def update_image_gen(n_clicks):
    image_bw ,image_net = execute_model()
    return ['data:image/png;base64,{}'.format(image_bw),
    'data:image/png;base64,{}'.format(image_net)]

@app.callback(
    [Output('graph-matrix-gen', 'src',allow_duplicate=True),
     Output('graph-image-gen', 'src',allow_duplicate=True)],
    Input('model-selector', 'value')
)
def update_model(selected_data):
    global model_default
    model_output_path = "./Microservices/Modelos/{}"
    model_default = joblib.load(model_output_path.format(selected_data))
    image_bw ,image_net = execute_model()
    return ['data:image/png;base64,{}'.format(image_bw),
    'data:image/png;base64,{}'.format(image_net)]

@app.callback(
    [Output('histogram-graph', 'figure'),
    Output('data-table', 'data'),
     Output('data-table', 'columns')],
    [Input('histogram-selector', 'value')]
)
def update_histogram(selected_data):
    selected_column = {
            "res_p" : res_load_p_columns ,
            "res_q": res_load_q_columns,
            "vm_pu": vm_columns,
            "va_degree":va_degree_columns,
        }
    select_columns_table = ["time_step"] + selected_column[selected_data]
    # select_columns_table.append("time_step")
    table_data = df_loads[select_columns_table].to_dict('records')
    columns = [{"name": i, "id": i} for i in df_loads[select_columns_table].columns]
    # fig = px.histogram(df_loads, x='time_step', y=selected_column[selected_data], histfunc='max')
    fig = px.line(df_loads, x='time_step', y=selected_column[selected_data])
    return [ fig ,table_data,columns]


@app.callback(
    [Output('dynamic-content', 'children'),
     Output('execute-button', 'style'),
     Output('graph-button', 'style')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_dynamic_content(contents, filename):
    global df_matrix
    global df_loads
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'load.csv' in filename:
                df_loads = pd.read_csv(StringIO(decoded.decode('utf-8')), header=[0])
                return (
                    html.Div([
                        html.Label('Modelo de Reconstrucción:', className='form-label'),
                        dcc.Dropdown(
                            id='model-selector',
                            options=[{'label': model, 'value': model} for model in model_names_files],
                            value=model_names_files[0],
                        )
                    ]),
                    {'display': 'block'},
                    {'display': 'none'}
                )
            elif 'incidence_matrix.csv' in filename:
                df_matrix = pd.read_csv(StringIO(decoded.decode('utf-8')), header=[0])
                df_matrix.rename(columns={"Unnamed: 0": "lines_id"}, inplace=True)
                df_matrix.set_index("lines_id", inplace=True)
                return (
                    html.Div(),
                    {'display': 'none'},
                    {'display': 'block'}
                )
        except Exception as e:
            return html.Div(['Hubo un error al procesar el archivo.']), {'display': 'none'}, {'display': 'none'}
    return html.Div(), {'display': 'none'}, {'display': 'none'}

if __name__ == '__main__':
    app.run_server(debug=True)
