import pandas as pd
import plotly.graph_objects as go
import scipy.stats as st
import numpy as np

def mosaic_chart(rc_table, title=None, residuals=False):
    labels = rc_table.columns
    heights = rc_table / rc_table.sum() * 100
    widths = rc_table.sum(axis=0) / rc_table.sum().sum() * 100
    expected = st.chi2_contingency(rc_table)[3]
    standartized_residuals = ((rc_table - expected) / expected ** .5)
    percentage_error = (heights.T - ((rc_table.sum(axis=1) / rc_table.sum().sum()) * 100)).T
    
    chart = []
    for i in heights.index:
        if residuals == 'standardized':
            marker = dict(cmin=-4, cmax=4, color=standartized_residuals.loc[i], colorbar={'title': ''},
                          colorscale='RdBu')
            customdata = [i] * len(labels)
            texttemplate = "%{customdata}: %{text:,}"
            showlegend = False
            
        elif residuals == 'percentage':
            cmax = int(percentage_error.max().max() // 10 + 1) * 10 
            marker = dict(cmin=-cmax, cmax=cmax, color=percentage_error.loc[i], colorbar={'title': ''},
                          colorscale='RdBu')
            customdata = [i] * len(labels)
            texttemplate = "%{customdata}: %{text:,}"
            showlegend = False
            
        else:
            marker = None
            customdata = None
            texttemplate = None
            showlegend = True

        h = heights.loc[i]
        chart += [go.Bar(y=h, x=np.cumsum(widths) - widths, width=widths, offset=0, name=i, textangle=0,
                         text=rc_table.loc[i], textposition='inside', marker=marker,
                         customdata=customdata, texttemplate=texttemplate,
                         hovertemplate="<br>".join(['height: %{y:.1f}%',
                                                    'width: %{width:.1f}%',
                                                    'value: %{text:,}'])
                         )]

    fig = go.Figure(chart)
    fig.update_layout(template='simple_white', barmode="stack", uniformtext={'mode': "hide", 'minsize': 12},
                      yaxis={'range': [0, 100], 'title':'percentage (%)'},
                      xaxis={'range': [0, 100], 'tickvals': np.cumsum(widths) - widths / 2,
                      'ticktext': ["{}<br>{:,}".format(l, w) for l, w in zip(labels, rc_table.sum(axis=0).tolist())]},
                      title={'text': title, 'x': .5}, showlegend=showlegend,
                      legend_title_text=rc_table.index.name)
    fig.show()
