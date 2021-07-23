import pandas as pd
import plotly.graph_objects as go


def pareto_chart(data):
    data = pd.Series(data)
    counts = (data.value_counts().to_frame('counts')
              .join(data.value_counts(normalize=True).cumsum().to_frame('ratio')))

    fig = go.Figure([go.Bar(x=counts.index, y=counts['counts'], yaxis='y1', name='Count'),
                     go.Scatter(x=counts.index, y=counts['ratio'], yaxis='y2', name='Cumulative Ratio',
                                hovertemplate='%{y:.2f}', marker={'color': '#000000'})])
    fig.update_layout(template='plotly_white',
                      hovermode='x',
                      title={'text': 'Pareto Chart', 'x': .5},
                      yaxis={'title': 'count'}, bargap=.3,
                      yaxis2={'rangemode': "tozero", 'overlaying': 'y',
                              'side': 'right', 'position': 1, 'title': 'ratio'})
    fig.show()
