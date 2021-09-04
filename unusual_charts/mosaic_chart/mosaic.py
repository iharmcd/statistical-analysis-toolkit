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

    chart = []
    for i in heights.index:
        if residuals:
            marker = dict(cmin=-4, cmax=4, color=standartized_residuals.loc[i], colorbar={'title': ''},
                          colorscale='RdBu')
            customdata = [i] * len(heights.index)
            texttemplate = "%{customdata}: %{text}"
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
                         hovertemplate="<br>".join(['height: %{y:.0f}%',
                                                    'width: %{width:.0f}%',
                                                    'value: %{text}'])
                         )]

    fig = go.Figure(chart)
    fig.update_layout(template='simple_white', barmode="stack", yaxis={'range': [0, 100]},
                      xaxis={'range': [0, 100], 'tickvals': np.cumsum(widths) - widths / 2,
                             'ticktext': ["%s<br>%d" % (l, w) for l, w in zip(labels, rc_table.sum(axis=0).tolist())]},
                      uniformtext={'mode': "hide", 'minsize': 12},
                      title={'text': title, 'x': .5}, showlegend=showlegend,
                      legend_title_text=rc_table.index.name)
    fig.show()
