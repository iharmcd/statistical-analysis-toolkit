#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as st
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
		html.Div(["Input number of messages: ", dcc.Input(id='trials', type='number', step=1, max=1001, min=0, value=10)]),
		html.Br(),
		html.Div(["Expected conversion rate: ", dcc.Input(id='probas',type='number', step=0.01, min=0, max=1, value=0.5)]),
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
def update_figures(trials, prob):

	def binominal(n, p):
		n_range = np.array(range(n+1))
		return n_range, st.binom.pmf(n_range, n, p)

	bar_chart = []	
	description = ''

	try:
		trials_range, binom_pmf = binominal(trials,prob)
		bar_chart = [(go.Bar(x=trials_range,y=binom_pmf, hoverinfo='text+x', text=list(map(lambda x: '{:.1%}'.format(x),binom_pmf)), marker_color='#FFA15A',
			opacity=0.85))]

		mx, var, _, _ = st.binom.stats(trials, prob, moments='mvsk')
	
		description = 'Mean ± Std: {:.0f} ± {:.3f}'.format(mx, var**.5)
	except:	
		pass

	return (
		{'data':  bar_chart, 
		'layout': go.Layout(template='plotly_white', hovermode='x', xaxis={'title':'possible outcomes'}, yaxis={'title':'probability'})},
		description,
	)

if __name__ == "__main__":
    app.run_server(host='0.0.0.0')






