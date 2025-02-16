import random
import base64
import joblib
import pandas as pd
from io import BytesIO ,StringIO


import plotly.express as px
import pandapower as pp
import pandapower.networks as pn
import matplotlib.pyplot as plt

from dash import Dash, html, dash_table, dcc
from dash.dependencies import Input, Output ,State

max_size_nodes = 24
nodes = 18
lines = nodes-1
loads = 9
num_intervals = 48
num_scenarios = 5
time_steps_simulation = num_intervals * 14

base_output_path = f"./Microservices/Data"
common_folder=f"{base_output_path}/train_data/AV3/case_net_generate_n_22_scenario_11"
loads_files=f"{common_folder}/load/load.csv"
matrix_file=f"{common_folder}/indicen_matriz/incidence_matrix.csv" 
df_matrix = pd.read_csv(matrix_file ,header=[0])
df_matrix.rename(columns={"Unnamed: 0":"lines_id"},inplace=True)
df_matrix.set_index("lines_id",inplace=True)
df_loads = pd.read_csv(loads_files ,header=[0])

select_columns_table = df_loads.columns
res_load_p_columns = [column for column in df_loads.columns if 'res_p' in column]
res_load_q_columns = [column for column in df_loads.columns if 'res_q' in column]
vm_columns = [column for column in df_loads.columns if 'vm' in column]
va_degree_columns = [column for column in df_loads.columns if 'va_degree' in column]
external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css']


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
    fig, ax = plt.subplots()
    pp.plotting.simple_plot(net, ax=ax)
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return image_base64


def generate_image():
    net = create_network(random.randint(0,3))
    slack_bus = 0
    pp.create_ext_grid(net, bus=slack_bus, vm_pu=1.02)
    fig, ax = plt.subplots()
    pp.plotting.simple_plot(net, ax=ax)
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return image_base64


# model_output_path ="./Modelos/{}"
# model_cargado5 = joblib.load(model_output_path.format("model_lstm_cnn-mask-zeros-001D"))


app = Dash(__name__ ,external_stylesheets=external_stylesheets) # 
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
                html.Div(className='col-md-4', children=[
                html.Label('Modelo de Reconstruccion:', className='form-label'),
                dcc.Dropdown(
                    id='model-selector',
                    options=[
                        {'label': 'Model A', 'value': 'res_p'},
                        {'label': 'Model B', 'value': 'res_q'},
                        {'label': 'Model C', 'value': 'vm_pu'},
                        {'label': 'Model D', 'value': 'va_degree'}
                    ],
                    value='res_p',
                ),
            html.Button('Ejecutar', id='update-form-button', n_clicks=0, className='btn btn-success mt-3'),
            html.Div(id='output-form', className='mt-6', style={'padding': '20px'}),
            ]),
        ]),
        html.Div(className='col-md-4', children=[
            html.Div(className='form-group', children=[
                html.Label('Numero de Nodos:', className='form-label'),
                dcc.Input(id='input-integer-1', type='number', value=0, className='form-control'),
            ]),
            html.Div(className='form-group', children=[
                html.Label('Numero de Cargas:', className='form-label'),
                dcc.Input(id='input-integer-2', type='number', value=0, className='form-control'),
            ]),
        ]),
    ]),
    html.Div(className='row', children=[
        html.Div(className='col-md-6', children=[
            html.H3('Red Original', className='text-center my-4'),
            html.Img(id='graph-image', className='img-fluid', style={'padding': '20px'}),
            html.Button('Actualizar Imagen', id='update-button', n_clicks=0, className='btn btn-primary mb-3'),
        ]),
        html.Div(className='col-md-6', children=[
            html.H3('Red Generada', className='text-center my-4'),
            html.Img(id='graph-image-2', className='img-fluid', style={'padding': '20px'}),
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

# @app.callback(
#     Output('output-form', 'children'),
#     [Input('update-form-button', 'n_clicks')],
#     [State('input-integer-1', 'value'),
#      State('input-integer-2', 'value'),
#      State('input-text-1', 'value'),
#      State('input-text-2', 'value'),
#      State('input-text-3', 'value')]
# )
# def update_form(n_clicks, int1, int2, text1, text2, text3):
#     return html.Div([
#         html.P(f'Valor Entero 1dftgg: {int1}'),
#         html.P(f'Valor Entero 2: {int2}'),
#         html.P(f'Texto 1: {text1}'),
#         html.P(f'Texto 2: {text2}'),
#         html.P(f'Texto 3: {text3}')
#     ])

@app.callback(
    [Output('graph-image', 'src'),
    Output('graph-image-2', 'src')],
    [Input('update-button', 'n_clicks')]
)
def update_image(n_clicks):
    image_base64 = plot_simple_df_net(df_matrix)

    return ['data:image/png;base64,{}'.format(image_base64),
    'data:image/png;base64,{}'.format(image_base64)]

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
    Output('output-form', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified')
)
def update_output(contents, filename, last_modified):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in filename:
                # Asumiendo que el archivo subido es un CSV
                global df_loads # = pd.read_csv(StringIO(decoded.decode('utf-8')),header=[0])
                df_loads = pd.read_csv(StringIO(decoded.decode('utf-8')),header=[0])
            elif 'xls' in filename:
                # Asumiendo que el archivo subido es un Excel
                df = pd.read_excel(BytesIO(decoded))
            return html.Div([
                    'el archivo fue procesado.'
                ])
            
            # [ html.Div([
            #     html.H5(filename),
            #     dash_table.DataTable(
            #         data=df.to_dict('records'),
            #         columns=[{'name': i, 'id': i} for i in df.columns]
            #     )
            # ]) , df_loads]
        except Exception as e:
            return html.Div([
                'Hubo un error al procesar el archivo.'
            ])

if __name__ == '__main__':
    app.run(debug=True)
