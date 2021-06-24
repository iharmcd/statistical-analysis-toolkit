#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import binom
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, compress=False)

app.layout = html.Div(children=[

		html.Center(html.H2(children = 'Responce Estimation with Binomial Distribution')),
		html.Hr(),
		html.Div(["Input number of messages: ", dcc.Input(id='trials', type='number', step=1, max=1001, min=0)]),
		html.Br(),
		html.Div(["Expected conversion rate: ", dcc.Input(id='probas',type='number', step=0.01, min=0, max=1)]),
		html.Hr(),
		html.Div([dcc.Graph( id = 'binomial_chart')], className='twelve columns'),
		html.H6(html.Div(id='output-info', style={'width': '95%','float':'right'})),
])

@app.callback(
    [Output('binomial_chart', 'figure'),
	Output('output-info', 'children'),
    ],
    [Input('trials', 'value'),
    Input('probas', 'value'),
    ])
def update_figures(trials, probas):

	def binominal(n, p):
		distribution = np.array([binom(n, k) * p**k * (1-p)**(n-k) for k in range(0, n+1)])
		return distribution

	bar_chart = []	
	description = ''

	try:
		data, trials = binominal(trials,probas), np.array(range(trials+1))
		bar_chart = [(go.Bar(x=trials,y=data, hoverinfo='text+x', text=list(map(lambda x: '{:.1%}'.format(x),data)), marker_color='#FFA15A',
			opacity=0.85))]

		mx = round((trials*data).sum())
		sigma = round((mx*(1-probas))**.5,3)
	
		description = 'Mean ± Std: {} ± {}'.format(mx, sigma)
	except:	
		pass

	return (
		{'data':  bar_chart, 
		'layout': go.Layout(template='plotly_white', hovermode='x', xaxis={'title':'possible outcomes'}, yaxis={'title':'probability'})},
		description,
	)

if __name__ == "__main__":
    app.run_server()






