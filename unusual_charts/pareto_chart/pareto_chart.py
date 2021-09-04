import pandas as pd
import plotly.graph_objects as go


def pareto_chart(collection, title=None):
    collection = pd.Series(collection)
    counts = (collection.value_counts().to_frame('counts')
              .join(collection.value_counts(normalize=True).cumsum().to_frame('ratio')))

    fig = go.Figure([go.Bar(x=counts.index, y=counts['counts'], yaxis='y1', name='count'),
                     go.Scatter(x=counts.index, y=counts['ratio']*100, yaxis='y2', name='cumulative ratio',
                                hovertemplate='%{y:.1f}%', marker={'color': '#000000'})])

    fig.update_layout(template='plotly_white', showlegend=False, hovermode='x', bargap=.3,
                      title={'text': title, 'x': .5}, 
                      yaxis={'title': 'count'},
                      yaxis2={'rangemode': "tozero", 'overlaying': 'y',
                              'position': 1, 'side': 'right',
                              'title': 'percantage (%)',
                              'range':[0,110]})

    fig.show()
