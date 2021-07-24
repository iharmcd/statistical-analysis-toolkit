import pandas as pd
import plotly.graph_objects as go


def pareto_chart(collection):
    collection = pd.Series(collection)
    counts = (collection.value_counts().to_frame('counts')
              .join(collection.value_counts(normalize=True).cumsum().to_frame('ratio')))

    fig = go.Figure([go.Bar(x=counts.index, y=counts['counts'], yaxis='y1', name=''),
                     go.Scatter(x=counts.index, y=counts['ratio'], yaxis='y2', name='',
                                hovertemplate='%{y:.1%}', marker={'color': '#000000'})])

    fig.update_layout(template='plotly_white', showlegend=False, hovermode='x', bargap=.3,
                      title={'text': 'Pareto Chart', 'x': .5}, 
                      yaxis={'title': 'count'},
                      yaxis2={'rangemode': "tozero", 'overlaying': 'y',
                              'position': 1, 'side': 'right',
                              'title': 'cumulative ratio',
                              'tickvals': np.arange(0, 1.1, .2),
                              'tickmode': 'array',
                              'ticktext': [str(i) + '%' for i in range(0, 101, 20)]})

    fig.show()
