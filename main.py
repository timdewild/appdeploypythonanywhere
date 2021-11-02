import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash import dash_table
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np


##############################################

def minmax(arr):
    minval = np.amin(arr)-0.5
    maxval = np.amax(arr)+0.5
    return [minval, maxval]

def x_uni_acc(t):
    return 0.5*t**2

def x_uni_vel(t):
    return 2*t



fig_xt = make_subplots(rows=1, cols=3)

fig_xt.add_trace(
    go.Scatter(x=[1, 2, 3], y=[1, 2, 3]),
    row=1, col=1
)

fig_xt.add_trace(
    go.Scatter(x=[1, 2, 3], y=[1, 2, 3]),
    row=1, col=2
)

fig_xt.add_trace(
    go.Scatter(x=[1, 2, 3], y=[1, 2, 3]),
    row=1, col=3
)

fig_xt.update_layout(height=400, width=1000, title_text="Position-Time Plots")

##############################################
#define the header card

header_card = dbc.Card(
    [
        dbc.CardHeader(
            html.H1("Motion Diagrams", className="card-title")
        ),
        dbc.CardBody(
            [
                html.P("This app checks your understanding of motion diagrams, and in particular how to convert them into corresponding velocity-time and position-time graphs."),
                html.Li("Select a motion diagram from the dropdown, which is shown to the right. You can choose to show velocity/acceleration vectors as well."),
                html.Li("Answer questions 1 and 2 (left panels) based on the given options (right panels)."),
                html.Li("Check your answer using the Check-button."),
                html.Li("Proceed to the next motion diagram.")
            ]
        ),
    ]
)

# define my first two cards

first_card = dbc.Card(
    [
    dbc.CardHeader(
        html.H4("Select a Diagram")
    ),

    dbc.CardBody(
            [
                dcc.Dropdown(
                    id='diagram-dd',
                    options=[
                        # {'label': 'Diagram {}'.format(x), 'value': x} for x in [1, 2, 3, 4],
                        {'label': 'Uniform Motion', 'value': 1}, {'label': 'Uniform Acceleration', 'value': 2}, {'label': 'Increasing Acceleration', 'value': 3}, {'label': 'Push Motion', 'value': 4},
                    ],
                    placeholder='Select a motion diagram',
                    multi=False,
                    value=1,
                    clearable=False,
                    style={"margin-bottom": "10px"}
                ),
                html.Hr(),
                html.H5("Show vectors:"),
                dcc.Checklist(
                    id='vel-acc-checklist',
                    options=[
                        {'label': 'Velocity', 'value': 'Velocity'}, {'label': 'Acceleration', 'value': 'Acceleration'}
                    ],
                    value=[],
                    labelStyle={'display': 'inline-block', 'margin-right':'10px'},
                    inputStyle={"margin-right": "5px"}
                ),

            ]
    ),
    ],
    style={"height": "30vh"},
)

second_card = dbc.Card(
    [
    dbc.CardHeader(
        html.H4(children={}, id='diagram-header')
    ),
    dbc.CardBody(
        [
            dcc.Graph(id='motion-diagram', figure={}),
        ]
    ),
    ],
    style={"height": "30vh"},
)

radio_group = dcc.RadioItems(
                    id='radio-ans',
                    options=[
                        {"label": "Left", "value": 1},
                        {"label": "Middle", "value": 2},
                        {"label": "Right", "value": 3},
                    ],
                    value=2,
                    labelStyle={'display': 'inline-block', 'margin-right':'10px'},
                    inputStyle={"margin-right": "5px"}
                )

radio_group_2 = dcc.RadioItems(
                    id='radio-ans-2',
                    options=[
                        {"label": "Left", "value": 1},
                        {"label": "Middle", "value": 2},
                        {"label": "Right", "value": 3},
                    ],
                    value=2,
                    labelStyle={'display': 'inline-block', 'margin-right':'10px'},
                    inputStyle={"margin-right": "5px"}
                )


third_card = dbc.Card(
    [
        dbc.CardHeader(
                html.H4("Question 1")
            ),
        dbc.CardBody(
                [
                    html.P("Which velocity-time diagram corresponds to the motion diagram?"),
                    radio_group,
                    dbc.Button("Check", color="secondary", style={"margin-top": "15px"}, n_clicks=0, id='answer-btn'),
                    dbc.Alert(
                        children="Correct",
                        id='alert-auto',
                        is_open=False,
                        duration=2000,
                        color="success",
                        style={"margin-top": "15px"}
                    ),
                ]
        )
    ],
    # style={"height": 350},
)

four_card = dbc.Card(
    [
        dbc.CardHeader(
            html.H4("Velocity-Time Graphs")
        ),
        dbc.CardBody(
            [
                dcc.Graph(id='vt-diagram', figure={}),
            ]
        ),
    ],
    # style={"height": 350},
)

fifth_card = dbc.Card(
    [
        dbc.CardHeader(
                html.H4("Question 2")
            ),
        dbc.CardBody(
                [
                    html.P("Which position-time diagram corresponds to the motion diagram?"),
                    radio_group_2,
                    dbc.Button("Check", color="secondary", style={"margin-top": "15px"}, n_clicks=0, id='answer-btn-2'),
                    dbc.Alert(
                        children="Correct",
                        id='alert-auto-2',
                        is_open=False,
                        duration=2000,
                        color="success",
                        style={"margin-top": "15px"}
                    ),
                ]
        )
    ],
    # style={"height": 350},
)

six_card = dbc.Card(
    [
        dbc.CardHeader(
            html.H4("Position-Time Graphs")
        ),
        dbc.CardBody(
            [
                dcc.Graph(id='xt-diagram', figure={}),
            ]
        ),
    ],
    # style={"height": 350},
)


##############################################

# initialize the app
app = dash.Dash()

app.layout = html.Div([
    html.Br(),
    dbc.Container([
    dbc.Row(
        [dbc.Col(header_card)],
        className="mb-4"
    ),

    dbc.Row(
        [
            dbc.Col(first_card, width=3),
            dbc.Col(second_card, width=9),
        ],
        className="mb-4"
    ),

    dbc.Row(
        [
            dbc.Col(third_card, width=3),
            dbc.Col(four_card, width=9),
        ],
        className="mb-4"
    ),

    dbc.Row(
        [
            dbc.Col(fifth_card, width=3),
            dbc.Col(six_card, width=9),
        ],
        className="mb-4"
    )

    ],
    style={"height": "100vh"},
    ),
    html.Br(),
    html.Br(),

])

# diagram header
@app.callback(
    Output('diagram-header', 'children'),
    Input('diagram-dd','value')
)
def update_header(n_diagram):
    if n_diagram == 1:
        hd_text = 'Motion Diagram: Uniform Motion'
    if n_diagram == 2:
        hd_text = 'Motion Diagram: Uniform Acceleration'
    if n_diagram == 3:
        hd_text = 'Motion Diagram: Increasing Acceleration'
    if n_diagram == 4:
        hd_text = 'Motion Diagram: Push Motion'
    return hd_text

# update of answers

@app.callback(
    [Output('alert-auto', 'is_open'), Output('alert-auto', 'children'), Output('alert-auto', 'color')],
    [Input('answer-btn', 'n_clicks')],
    [State('alert-auto', 'is_open'), State('radio-ans', 'value'), State('diagram-dd','value')]
)
def update_answer(n_button, is_open, u_answer, n_diagram):
    c_answers = [None, 1,3,3,2]
    if n_button:
        isopen = not is_open
    else:
        isopen = is_open

    if u_answer == c_answers[n_diagram]:
        if n_diagram == 1:
            alert_text = "Correct! This is an example of constant velocity motion."
        if n_diagram == 2:
            alert_text = "Correct! This is an example of constant acceleration motion."
        if n_diagram == 3:
            alert_text = "Correct! This is an example of increasing acceleration."
        if n_diagram == 4:
            alert_text = "Correct! This is an example of (almost) instantaneous acceleration."
        alert_col = "success"

    else:
        alert_text = "Incorrect!"
        alert_col = "danger"

    return isopen, alert_text, alert_col

@app.callback(
    [Output('alert-auto-2', 'is_open'), Output('alert-auto-2', 'children'), Output('alert-auto-2', 'color')],
    [Input('answer-btn-2', 'n_clicks')],
    [State('alert-auto-2', 'is_open'), State('radio-ans-2', 'value'), State('diagram-dd','value')]
)
def update_answer_2(n_button, is_open, u_answer, n_diagram):
    c_answers = [None, 2,1,2,1]
    if n_button:
        isopen = not is_open
    else:
        isopen = is_open

    if u_answer == c_answers[n_diagram]:
        if n_diagram == 1:
            alert_text = "Correct! This is an example of constant velocity motion."
        if n_diagram == 2:
            alert_text = "Correct! This is an example of constant acceleration motion."
        if n_diagram == 3:
            alert_text = "Correct! This is an example of increasing acceleration."
        if n_diagram == 4:
            alert_text = "Correct! This is an example of (almost) instantaneous acceleration."
        alert_col = "success"

    else:
        alert_text = "Incorrect!"
        alert_col = "danger"

    return isopen, alert_text, alert_col

#######################################################################################################################

# motion diagram overall layout
layout_md = {'xaxis': {'title': 'x',
                       'visible': True,
                       'showticklabels': True,
                       'ticks': 'outside',
                       'ticklen': 10,
                       'tickmode': 'linear',
                       'tick0': 0.,
                       'dtick': 1.,
                       },
              'yaxis': {'title': 'y-label',
                        'visible': False,
                        'showticklabels': True,
                        'range': [1.5,2.5]
                        },
              #'width': 1000,
              'height': 150,
              'margin': {'t': 10, 'r': 10, 'b': 50, 'l': 10}
              }

def arrow(xb,yb,xe,ye,color = 'black', xref='x', yref='y'):
    arr = {
        'x': xe,
        'y': ye,
        'ax': xb,
        'ay': yb,
        'xref': xref,
        'yref': yref,
        'axref': xref,
        'ayref': yref,
        'text': '',
        'showarrow': True,
        'arrowhead': 2,
        'arrowsize': 1,
        'arrowwidth': 2,
        'arrowcolor': color
    }

    return arr

def plotter(t,x):
    return 0


tvals = np.arange(0,6,1)
yvals = 2*np.ones_like(tvals)

def x_gen(n):
    if n == 1:
        x = x_uni_vel(tvals)
        y = 2*np.ones_like(x)
        return x, minmax(x), y
    if n == 2:
        x = x_uni_acc(tvals)
        y = 2 * np.ones_like(x)
        return x, minmax(x), y
    if n == 3:
        x = [0,1,4,10,20,35]
        y = 2 * np.ones_like(x)
        return x, minmax(x), y
    if n == 4:
        x = [-4,-3,-2,-1,0,2,4,6]
        y = 2 * np.ones_like(x)
        return x, minmax(x), y
    else:
        return np.zeros_like(tvals)

xuni=x_uni_vel(tvals)
xuni_b = minmax(xuni)

def vel_acc_gen(n):
    x, xb, y = x_gen(n)
    xe = x[1:]
    xs = x[:-1]
    v = np.transpose([xs,xe]).tolist()

    print(v)

    vlist = []
    for i in range(np.size(x)-1):
        vlist.append(arrow(v[i][0],2,v[i][1],2,'red'))

    Alist = []
    for i in range(1,np.size(x)-1):
        As = v[i][0]
        v1 = v[i-1][1]-v[i-1][0]
        v2 = v[i][1]-v[i][0]
        Ae = As+v2-v1
        Alist.append(arrow(As,2,Ae,2,'green'))


    print(vlist)
    print(Alist)

    return vlist, Alist

vel_acc_gen(1)

def fig_gen(n):
    # boundary values x
    xvals, xb, y = x_gen(n)
    xmin = xb[0]; xmax = xb[1]

    # list with arrows
    arr_list = [arrow(xmin, 1.5, xmax, 1.5)]

    # v and a arrows
    vlist, alist = vel_acc_gen(n)

    # generate figure object
    fig = go.Figure(layout=layout_md)
    fig.add_scatter(x=xvals, y=y, mode='markers')
    fig.update_xaxes(range=xb)
    fig.update_layout({'annotations': arr_list})

    # generate figure with velocity arrows
    fig_v = go.Figure(layout=layout_md)
    fig_v.add_scatter(x=xvals, y=y, mode='markers')
    fig_v.update_xaxes(range=xb)
    fig_v.update_layout({'annotations': arr_list+vlist})

    # generate figure with acceleration arrows
    fig_a = go.Figure(layout=layout_md)
    fig_a.add_scatter(x=xvals, y=y, mode='markers')
    fig_a.update_xaxes(range=xb)
    fig_a.update_layout({'annotations': arr_list + alist})

    # generate figure with velocity and acceleration arrows
    fig_va = go.Figure(layout=layout_md)
    fig_va.add_scatter(x=xvals, y=y, mode='markers')
    fig_va.update_xaxes(range=xb)
    fig_va.update_layout({'annotations': arr_list + vlist + alist})

    return fig, fig_v, fig_a, fig_va



@app.callback(
    Output(component_id='motion-diagram', component_property='figure'),
    [Input(component_id='diagram-dd', component_property='value'),
     Input(component_id='vel-acc-checklist', component_property='value')]
)
def update_motion_diagram(n_diagram,checklist):
    fig, fig_v, fig_a, fig_va = fig_gen(n_diagram)

    if checklist == []:
        return fig

    if checklist == ['Velocity']:
        return fig_v

    if checklist == ['Acceleration']:
        return fig_a

    if checklist == ['Velocity', 'Acceleration'] or checklist == ['Acceleration', 'Velocity']:
        return fig_va

def vt_data(n_diagram):
    if n_diagram == 1:
        y1 = [2, 2, 2, 2, 2, 2]
        y2 = [0.5, 1, 1.5, 2, 2.5, 3]
        y3 = [-2, -2, -2, -2, -2, -2]
        t1 = [0, 1, 2, 3, 4, 5]
        t2 = [0, 1, 2, 3, 4, 5]
        t3 = [0, 1, 2, 3, 4, 5]
        tbounds1 = minmax(t1)
        tbounds2 = minmax(t2)
        tbounds3 = minmax(t3)

        return y1, y2, y3, t1, t2, t3, tbounds1, tbounds2, tbounds3

    if n_diagram == 2:
        y1 = [2.5, 1.5, 0.5, -0.5, -1.5, -2.5]
        y2 = [-2.5,-1.5,-0.5,0.5,1.5,2.5]
        y3 = [0.5, 1, 1.5, 2, 2.5, 3]
        t1 = [0, 1, 2, 3, 4, 5]
        t2 = [0, 1, 2, 3, 4, 5]
        t3 = [0, 1, 2, 3, 4, 5]
        tbounds1 = minmax(t1)
        tbounds2 = minmax(t2)
        tbounds3 = minmax(t3)

        return y1, y2, y3, t1, t2, t3, tbounds1, tbounds2, tbounds3

    if n_diagram == 3:
        y1 = [-2.5,-1.5,-0.5,0.5,1.5,2.5]
        y2 = np.array([0,1,4,9,16,25])*-3/25
        y3 = np.array([0,1,4,9,16,25])*3/25
        t1 = [0, 1, 2, 3, 4, 5]
        t2 = [0, 1, 2, 3, 4, 5]
        t3 = [0, 1, 2, 3, 4, 5]
        tbounds1 = minmax(t1)
        tbounds2 = minmax(t2)
        tbounds3 = minmax(t3)

        return y1, y2, y3, t1, t2, t3, tbounds1, tbounds2, tbounds3

    if n_diagram == 4:
        t1 = [0, 1, 2, 3, 4, 5, 6, 7]
        t2 = [0, 1, 2, 3, 4, 5, 6, 7]
        t3 = [0, 1, 2, 3, 4, 5, 6, 7]
        y1 = [2, 2, 2, 2, 1, 1, 1, 1]
        y2 = [1, 1, 1, 1, 1, 2, 2, 2]
        y3 = [-1, -1, -1, -1, 1, 1, 1, 1]
        tbounds1 = minmax(t1)
        tbounds2 = minmax(t2)
        tbounds3 = minmax(t3)

        return y1, y2, y3, t1, t2, t3, tbounds1, tbounds2, tbounds3


    else:
        y1 = [2, 2, 2, 2, 2, 2]
        y2 = [0.5, 1, 1.5, 2, 2.5, 3]
        y3 = [-2, -2, -2, -2, -2, -2]
        t1 = [0, 1, 2, 3, 4, 5]
        t2 = [0, 1, 2, 3, 4, 5]
        t3 = [0, 1, 2, 3, 4, 5]
        tbounds1 = minmax(t1)
        tbounds2 = minmax(t2)
        tbounds3 = minmax(t3)

        return y1, y2, y3, t1, t2, t3, tbounds1, tbounds2, tbounds3


def xt_data(n_diagram):
    if n_diagram == 1:
        x1 = [2,2,2,2,2,2]
        x2 = [0,0.5,1,1.5,2,2.5]
        x3 = [-0.5,0,0.5,1,1.5,2,]
        T1 = [0,1,2,3,4,5]
        T2 = [0,1,2,3,4,5]
        T3 = [0,1,2,3,4,5]
        Tbounds1 = minmax(T1)
        Tbounds2 = minmax(T2)
        Tbounds3 = minmax(T3)

        return x1, x2, x3, T1, T2, T3, Tbounds1, Tbounds2, Tbounds3

    if n_diagram == 2:
        x1 = np.array([0,1,4,9,16,25])*3/25
        x2 = np.array([9,16,25,36,49,64])*3/64
        x3 = np.array([0,1,4,9,16,25])*-3/25
        T1 = [0,1,2,3,4,5]
        T2 = [0,1,2,3,4,5]
        T3 = [0,1,2,3,4,5]
        Tbounds1 = minmax(T1)
        Tbounds2 = minmax(T2)
        Tbounds3 = minmax(T3)

        return x1, x2, x3, T1, T2, T3, Tbounds1, Tbounds2, Tbounds3

    if n_diagram == 3:
        x1 = np.array([0,1,2,3,4,5])**2*3/5**2
        x2 = np.array([0,1,2,3,4,5])**3*3/5**3
        x3 = np.array([0,1,2,3,4,5])*3/5
        T1 = [0,1,2,3,4,5]
        T2 = [0,1,2,3,4,5]
        T3 = [0,1,2,3,4,5]
        Tbounds1 = minmax(T1)
        Tbounds2 = minmax(T2)
        Tbounds3 = minmax(T3)

        return x1, x2, x3, T1, T2, T3, Tbounds1, Tbounds2, Tbounds3

    if n_diagram == 4:
        x1 = [-2,-1.5,-1,-0.5,0,1,2,3]
        x2 = [-3,-2,-1,0,0.75,1.5,2.25,3]
        x3 = [0, 0.2, 0.4, 0.6, 0.8,1,2,3]
        T1 = [0-4, 1-4, 2-4, 3-4, 4-4, 5-4, 6-4, 7-4]
        T2 = [0, 1, 2, 3, 4, 5, 6, 7]
        T3 = [0, 1, 2, 3, 4, 5, 6, 7]
        Tbounds1 = minmax(T1)
        Tbounds2 = minmax(T2)
        Tbounds3 = minmax(T3)

        return x1, x2, x3, T1, T2, T3, Tbounds1, Tbounds2, Tbounds3

    else:
        x1 = [2, 2, 2, 2, 2, 2]
        x2 = [0.5, 1, 1.5, 2, 2.5, 3]
        x3 = [-2, -2, -2, -2, -2, -2]
        T1 = [0, 1, 2, 3, 4, 5, 6, 7]
        T2 = [0, 1, 2, 3, 4, 5, 6, 7]
        T3 = [0, 1, 2, 3, 4, 5, 6, 7]
        Tbounds1 = minmax(T1)
        Tbounds2 = minmax(T2)
        Tbounds3 = minmax(T3)

        return x1, x2, x3, T1, T2, T3, Tbounds1, Tbounds2, Tbounds3

@app.callback(
    Output('xt-diagram','figure'),
    [Input('diagram-dd','value')]
)
def xt_gen(n_diagram):
    y1, y2, y3, t1, t2, t3, tbounds1, tbounds2, tbounds3 = xt_data(n_diagram)

    fig_xt = make_subplots(rows=1, cols=3)
    fig_xt.update_xaxes(range=tbounds1, dtick=1, row=1, col=1, showticklabels=False, title='t')
    fig_xt.update_xaxes(range=tbounds2, dtick=1, row=1, col=2, showticklabels=False, title='t')
    fig_xt.update_xaxes(range=tbounds3, dtick=1, row=1, col=3, showticklabels=False, title='t')
    fig_xt.update_yaxes(range=[-3.5, 3.5], dtick=1, showticklabels=False, title='x')

    fig_xt.add_trace(
        go.Scatter(x=t1, y=y1),
        row=1, col=1
    )

    fig_xt.add_trace(
        go.Scatter(x=t2, y=y2),
        row=1, col=2
    )

    fig_xt.add_trace(
        go.Scatter(x=t3, y=y3),
        row=1, col=3
    )

    fig_xt.update_layout(margin=dict(l=30, r=30, t=30, b=30), showlegend=False, height=275),

    fig_xt.add_annotation(arrow(tbounds1[0], 0, tbounds1[1], 0, 'gray', 'x1', 'y1'))
    fig_xt.add_annotation(arrow(tbounds2[0], 0, tbounds2[1], 0, 'gray', 'x2', 'y2'))
    fig_xt.add_annotation(arrow(tbounds3[0], 0, tbounds3[1], 0, 'gray', 'x3', 'y3'))

    fig_xt.add_annotation(arrow(0, -3.5, 0, 3.5, 'gray', 'x1', 'y1'))
    fig_xt.add_annotation(arrow(0, -3.5, 0, 3.5, 'gray', 'x2', 'y2'))
    fig_xt.add_annotation(arrow(0, -3.5, 0, 3.5, 'gray', 'x3', 'y3'))

    return fig_xt


@app.callback(
    Output('vt-diagram','figure'),
    [Input('diagram-dd','value')]
)
def vt_gen(n_diagram):
    y1, y2, y3, t1, t2, t3, tbounds1, tbounds2, tbounds3 = vt_data(n_diagram)

    fig_vt = make_subplots(rows=1, cols=3)
    fig_vt.update_xaxes(range=tbounds1, dtick=1, row=1, col=1, showticklabels=False, title='t')
    fig_vt.update_xaxes(range=tbounds2, dtick=1, row=1, col=2, showticklabels=False, title='t')
    fig_vt.update_xaxes(range=tbounds3, dtick=1, row=1, col=3, showticklabels=False, title='t')
    fig_vt.update_yaxes(range=[-3.5, 3.5], dtick=1, showticklabels=False,title='v<sub>x</sub>')

    fig_vt.add_trace(
        go.Scatter(x=t1, y=y1),
        row=1, col=1
    )

    fig_vt.add_trace(
        go.Scatter(x=t2, y=y2),
        row=1, col=2
    )

    fig_vt.add_trace(
        go.Scatter(x=t3, y=y3),
        row=1, col=3
    )

    fig_vt.update_layout(margin=dict(l=30, r=30, t=30, b=30), showlegend=False, height=275),

    fig_vt.add_annotation(arrow(tbounds1[0],0,tbounds1[1],0, 'gray', 'x1', 'y1'))
    fig_vt.add_annotation(arrow(tbounds2[0],0,tbounds2[1],0, 'gray', 'x2', 'y2'))
    fig_vt.add_annotation(arrow(tbounds3[0],0,tbounds3[1],0, 'gray', 'x3', 'y3'))

    fig_vt.add_annotation(arrow(0, -3.5, 0, 3.5, 'gray', 'x1', 'y1'))
    fig_vt.add_annotation(arrow(0, -3.5, 0, 3.5, 'gray', 'x2', 'y2'))
    fig_vt.add_annotation(arrow(0, -3.5, 0, 3.5, 'gray', 'x3', 'y3'))

    return fig_vt









#####################################################################################
if __name__ == '__main__':
    app.run_server(port=4052, host='127.0.0.1')

