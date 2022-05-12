from jupyter_dash import JupyterDash
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import pickle
import numpy as np
import os
import sys
import plotly.graph_objs as go

path = os.getcwd().split(os.sep +'GUI')[0]
if path not in sys.path:
    sys.path.append(path)

from neurolib.models.aln import ALNModel
from neurolib.dashboard import layout as layout
import neurolib.dashboard.functions as functions
import neurolib.dashboard.data as data

aln = ALNModel()
data.set_parameters(aln)

path = './'
readpath = path + 'data_final' + os.sep

##### LOAD BOUNDARIES
with open(path + 'boundary_bi_granular.pickle','rb') as file:
    load_array= pickle.load(file)
boundary_bi_exc = load_array[0]
boundary_bi_inh = load_array[1]

with open(path + 'boundary_LC_granular.pickle','rb') as file:
    load_array= pickle.load(file)
boundary_LC_exc = load_array[0]
boundary_LC_inh = load_array[1]

with open(path + 'boundary_LCbi_granular.pickle','rb') as file:
    load_array= pickle.load(file)
boundary_LC_up_exc = load_array[0]
boundary_LC_up_inh = load_array[1]

print('boundaries loaded')

tasks = ['LH1: low to high, L1 cost constraints',
         'LH2: low to high, L2 cost constraints',
         'HL1: high to low, L1 cost constraints',
         'HL2: high to low, L2 cost constraints']

global ind_, type_, mu_e, mu_i, a_e, a_i, cost_node, w_e, w_i, target_high, target_low
global bestControl_init, costnode_init, bestControl_0, bestState_0, costnode_0
global case

case = '1'

data_array = data.read_data_1(aln, readpath, case)
ind_, type_, mu_e, mu_i, a_e, a_i, cost_node, w_e, w_i, target_high, target_low = data_array
[bestControl_init, costnode_init, bestControl_0, bestState_0, costnode_0] = data.read_control(readpath, case)

print('read data case 1')

data1, data2, data4 = data.get_scatter_data_1(ind_, type_, mu_e, mu_i, a_e, a_i)
data_background = data.get_data_background(data1.x, data1.y, data2.x, data2.y, data4.x, data4.y)
trace00, trace01 = data.get_step_current_traces(aln)
trace10, trace11 = layout.get_empty_traces()

bistable_regime = layout.get_bistable_paths(boundary_bi_exc, boundary_bi_inh)
oscillatory_regime = layout.get_osc_path(boundary_LC_exc, boundary_LC_inh)
LC_up_regime = layout.get_LC_up_path(boundary_LC_up_exc, boundary_LC_up_inh)

fig_bifurcation = go.FigureWidget(data=[data_background, data1, data2, data4])#, data3, data4])
fig_bifurcation.update_layout(shapes=[bistable_regime, oscillatory_regime, LC_up_regime])
fig_bifurcation.add_annotation(layout.get_label_bistable())
fig_bifurcation.add_annotation(layout.get_label_osc())
fig_bifurcation.add_annotation(layout.get_label_osc_up())
fig_bifurcation.add_annotation(layout.get_label_down())
fig_bifurcation.add_annotation(layout.get_label_up())
fig_bifurcation.update_layout(layout.get_layout_bifurcation())

fig_time_series_exc = go.Figure(data=[trace00, trace10])
fig_time_series_exc.update_layout(layout.get_layout_exc())
fig_time_series_inh = go.Figure(data=[trace01, trace11])
fig_time_series_inh.update_layout(layout.get_layout_exc())
fig_time_series_inh.layout.title['text'] = 'Inhibitory node'

state0_e, state0_i = layout.get_empty_state()
cntrl0_e, cntrl0_i = layout.get_empty_control()

fig_opt_cntrl_exc = go.Figure(data=[state0_e, cntrl0_e])
fig_opt_cntrl_exc.update_layout(layout.get_layout_cntrl_exc())
fig_opt_cntrl_inh = go.Figure(data=[state0_i, cntrl0_i])
fig_opt_cntrl_inh.update_layout(layout.get_layout_cntrl_exc())
fig_opt_cntrl_inh.layout.yaxis['range'] = [0., 150.]
fig_opt_cntrl_inh.layout.yaxis2['range'] = [-1.6,0.2]
fig_opt_cntrl_inh.layout.title['text'] = 'Inhibitory node'

fig_tab_cost = go.Figure(
    data=go.Table(
        header=dict(values=['<b>Cost</b>', 'Excitatory', 'Inhibitory'], fill_color=layout.midgrey, align='center', font=dict(size=layout.text_fontsize), height=layout.text_fontsize+10),
        cells=dict(values=[['Precision', 'Sparsity', 'Energy'], [0., 0., 0.], [0., 0., 0.]], fill_color=[[layout.lightgrey, layout.lightgrey, layout.darkgrey]*3], font=dict(size=layout.text_fontsize), height=layout.text_fontsize+10)
        )
    )


case = '1'

app = JupyterDash(__name__)
app.layout = html.Div([
    html.H1("State switching task in bistable regime"),
    html.Div([
        html.H3("Task"),
        html.Label([
              dcc.Dropdown(
                  id='task_dropdown',
                  style={'font-size': '18px'},
                  clearable=False,
                  value=tasks[0],
                  options=[
                      {'label': t, 'value': t}
                      for t in tasks
                  ])
          ]),
        html.Br(),
        dcc.Graph(id='bifurcation_diagram',
                 figure=fig_bifurcation,
                 ),        
    ], style={'width': '26%', 'display': 'inline-block', 'padding': '0 0'}),
    html.Div([
        html.H3("Time series for stimulation with rectangle pulse"),
        html.Div([
            dcc.Graph(id='time_series_exc',
                 figure=fig_time_series_exc,
                 ),
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '0 0'}),
        html.Div([
            dcc.Graph(id='time_series_inh',
                 figure=fig_time_series_inh,
                 ),
            ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
        html.H3("Optimal transition and control"),
        html.Div([
            dcc.Graph(id='opt_cntrl_exc',
                 figure=fig_opt_cntrl_exc,
                 ),
            ], style={'width': '48%', 'display': 'inline-block', 'padding': '0 0'}),
        html.Div([
            dcc.Graph(id='opt_cntrl_inh',
                 figure=fig_opt_cntrl_inh,
                 ),
            ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
        #html.P(id='cost_output'),
    ], style={'width': '49%', 'float': 'right', 'display': 'inline-block', 'padding': '0 0'}),
    html.Div([
        dcc.Graph(id='tab_cost',
                figure=fig_tab_cost,
                ),
        ], style={'width': '25%', 'float': 'right', 'display': 'inline-block'}),
])


@app.callback(
        Output('bifurcation_diagram', 'figure'),
        Output('opt_cntrl_exc', 'figure'),
        Output('opt_cntrl_inh', 'figure'),
        Output('time_series_exc', 'figure'),
        Output('time_series_inh', 'figure'),
        Output('tab_cost', 'figure'),
        Input('bifurcation_diagram', 'clickData'),
        Input('task_dropdown', 'value'))
def set_marker(selection_click, selection_drop):
    
    if not dash.callback_context.triggered:
        return fig_bifurcation, fig_opt_cntrl_exc, fig_opt_cntrl_inh, fig_time_series_exc, fig_time_series_inh, fig_tab_cost
    
    global ind_, type_, mu_e, mu_i, a_e, a_i, cost_node, w_e, w_i, target_high, target_low, case
    global bestControl_init, costnode_init, bestControl_0, bestState_0, costnode_0
        
    if dash.callback_context.triggered[0]['prop_id'] == 'bifurcation_diagram.clickData':
        pInd = selection_click['points'][0]['pointIndex']
        trace = fig_bifurcation.data[selection_click['points'][0]['curveNumber']]

        functions.setdefaultmarkersize(0, fig_bifurcation.data[0])
        for fig_ in fig_bifurcation.data[1:]:
            functions.setdefaultmarkersize(layout.markersize, fig_)
        functions.setmarkersize(pInd, layout.background_markersize, trace)
        
        if selection_click['points'][0]['curveNumber'] == 0:
            data.set_opt_cntrl_plot_zero(fig_opt_cntrl_exc, [0,1])
            data.set_opt_cntrl_plot_zero(fig_opt_cntrl_inh, [0,1])
            fig_tab_cost.data[0]['cells']['values'] = [['Precision', 'Sparsity', 'Energy'], [0., 0., 0.], [0., 0., 0.]]
        
        else:        
            for i in range(len(ind_)):
                if (np.abs(mu_e[i] - selection_click['points'][0]['x']) < 1e-6
                and np.abs(mu_i[i] - selection_click['points'][0]['y']) < 1e-6):
                    index_ = ind_[i]

            time_ = np.arange(0., layout.simulation_duration + aln.params.dt, aln.params.dt)
        
            fig_opt_cntrl_exc.data[0].x = time_
            fig_opt_cntrl_exc.data[0].y = bestState_0[index_][0,0,:]
            fig_opt_cntrl_exc.data[1].x = time_
            fig_opt_cntrl_exc.data[1].y = bestControl_0[index_][0,0,:]
        
            fig_opt_cntrl_inh.data[0].x = time_
            fig_opt_cntrl_inh.data[0].y = bestState_0[index_][0,1,:]
            fig_opt_cntrl_inh.data[1].x = time_
            fig_opt_cntrl_inh.data[1].y = bestControl_0[index_][0,1,:]

            d_cost = [['Precision', 'Sparsity', 'Energy'], [0., 0., 0.], [0., 0., 0.]]
            
            c_p = costnode_0[index_][0][0,:]
            d_cost[1][0] = round(c_p[0], 4)
            d_cost[2][0] = round(c_p[1], 4)
            c_s = costnode_0[index_][2][0,:]
            d_cost[1][1] = round(c_s[0], 4)
            d_cost[2][1] = round(c_s[1], 4)
            c_e = costnode_0[index_][1][0,:]
            d_cost[1][2] = round(c_e[0], 4)
            d_cost[2][2] = round(c_e[1], 4)
            
            fig_tab_cost.data[0]['cells']['values'] = d_cost
            print("set tab cost ", d_cost, case)
            
        time_, exc_trace_, inh_trace_ = data.trace_step(aln, selection_click['points'][0]['x'],
                                                        selection_click['points'][0]['y'])

        fig_time_series_exc.data[1].x = time_
        fig_time_series_exc.data[1].y = exc_trace_ 
        fig_time_series_inh.data[1].x = time_
        fig_time_series_inh.data[1].y = inh_trace_
    
    elif dash.callback_context.triggered[0]['prop_id'] == 'task_dropdown.value':
        value = dash.callback_context.triggered[0]['value']
        
        case0 = case
        
        for i in range(len(tasks)):
            if tasks[i] == value:
                case = str(i+1)
                
        functions.setdefaultmarkersize(0, fig_bifurcation.data[0])
        for fig_ in fig_bifurcation.data[1:]:
            functions.setdefaultmarkersize(layout.markersize, fig_)
            
        for fig, i_list in zip([fig_opt_cntrl_exc, fig_opt_cntrl_inh, fig_time_series_exc, fig_time_series_inh], [[0,1], [0,1], [1], [1]]):
            data.set_opt_cntrl_plot_zero(fig, i_list)   
        
        data_array = data.read_data_1(aln, readpath, case)
        ind_, type_, mu_e, mu_i, a_e, a_i, cost_node, w_e, w_i, target_high, target_low = data_array
        [bestControl_init, costnode_init, bestControl_0, bestState_0, costnode_0] = data.read_control(readpath, case)
        
        data1, data2, data4 = data.get_scatter_data_1(ind_, type_, mu_e, mu_i, a_e, a_i)
        data_background = data.get_data_background(data1.x, data1.y, data2.x, data2.y, data4.x, data4.y)
        
        for (j, data_) in zip(range(3), [data_background, data1, data2]):
            data.set_data(fig_bifurcation, j, data_)
        
        if case0 in ['1', '2'] and case in ['3', '4']:
            fig_opt_cntrl_exc.layout.yaxis2['range'] = [-2.2,0.2]
            fig_opt_cntrl_inh.layout.yaxis2['range'] = [-0.4,0.4]
        elif case0 in ['3', '4'] and case in ['1', '2']:
            fig_opt_cntrl_exc.layout.yaxis2['range'] = [-0.2,2.]
            fig_opt_cntrl_inh.layout.yaxis2['range'] = [-1.6,0.2]

        if case in ['1', '3']:
            color_array = [[layout.lightgrey, layout.lightgrey, layout.darkgrey]*3]
        else:
            color_array = [[layout.lightgrey, layout.darkgrey, layout.lightgrey]*3]
            
        fig_tab_cost.data[0]['cells']['values'] = [['Precision', 'Sparsity', 'Energy'], [0., 0., 0.], [0., 0., 0.]]
        fig_tab_cost.data[0]['cells']['fill_color'] = color_array
            
    return fig_bifurcation, fig_opt_cntrl_exc, fig_opt_cntrl_inh, fig_time_series_exc, fig_time_series_inh, fig_tab_cost


app.run_server()