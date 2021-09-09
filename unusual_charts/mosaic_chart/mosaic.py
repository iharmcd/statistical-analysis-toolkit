import pandas as pd
import plotly.graph_objects as go
import scipy.stats as st
import numpy as np

def mosaic_chart(rc_table, title=None, residuals=None):
    labels = rc_table.columns
    heights = rc_table / rc_table.sum() 
    widths = rc_table.sum(axis=0) / rc_table.sum().sum()
    expected = st.chi2_contingency(rc_table)[3]
    standartized_residuals = ((rc_table - expected) / expected ** .5)
    percentage_error = heights - expected / expected.sum(axis=0)
    
    chart = []
    for i in heights.index:
        if residuals == 'standardized':
            marker = dict(cmin=-4, cmax=4, color=standartized_residuals.loc[i], colorscale='RdBu',
                          colorbar={'title': ''})
            customdata = [i] * len(labels)
            texttemplate = "%{customdata}: %{text:,}"
            showlegend = False
            
        elif residuals == 'percentage':
            marker = dict(cmin=-100, cmax=100, color=percentage_error.loc[i]*100, colorscale='edge_r',
                          colorbar={'ticktext':list(range(-100,101,20)), 
                                    'tickvals':list(range(-100,101,20)), 'title': ''})
            customdata = [i] * len(labels)
            texttemplate = "%{customdata}: %{text:,}"
            showlegend = False
            
        elif residuals is None:
            marker = None
            customdata = None
            texttemplate = None
            showlegend = True
        
        else:
            raise ValueError(f"Invalid property name.\
            \nRecieved value: '{residuals}' \n\nUse ['standardized', 'percentage', None].")

        h = heights.loc[i]
        chart += [go.Bar(y=h*100, x=(np.cumsum(widths) - widths)*100, width=widths*100, offset=0, 
                         text=rc_table.loc[i], textposition='inside', marker=marker, name=i, textangle=0,
                         customdata=customdata, texttemplate=texttemplate,
                         hovertemplate="<br>".join(['height: %{y:.1f}%',
                                                    'width: %{width:.1f}%',
                                                    'value: %{text:,}'])
                         )]

    fig = go.Figure(chart)
    fig.update_layout(template='simple_white', barmode="stack", uniformtext={'mode': "hide", 'minsize': 12},
                      yaxis={'range': [0, 100], 'title':'percentage (%)'},
                      xaxis={'range': [0, 100], 'tickvals': (np.cumsum(widths) - widths / 2) * 100,
                'ticktext': ["{}<br>{:,}".format(l, w) for l, w in zip(labels, rc_table.sum(axis=0).tolist())]},
                      title={'text': title, 'x': .5}, showlegend=showlegend,
                      legend_title_text=rc_table.index.name)
    fig.show()
