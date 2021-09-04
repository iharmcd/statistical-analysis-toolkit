# Mosaic Chart 
Mosaic (Marimekko) chart using python and plotly, for categorical variables analysis.
It is the same chart that [statsmodels provides](https://www.statsmodels.org/stable/generated/statsmodels.graphics.mosaicplot.mosaic.html).
> Create a mosaic plot from a contingency table.
>
> It allows to visualize multivariate categorical data in a rigorous and informative way.


Example of simple Marimekko chart:
<img width="1440" alt="Снимок экрана 2021-09-04 в 16 17 29" src="https://user-images.githubusercontent.com/7578492/132095857-04c3671d-e478-484f-847f-92e6d15e59a8.png">

If you want to explore data with standardized residuals, you can create mosaic chart, using `residuals=True`




> According to the https://www.cedu.niu.edu/~walker/statistics/Chi%20Square%202.pdf:
> 
> Standardized residual = O - E / √E
> 
> The standardized residual can be interpreted as any standard score. The mean of the
standardized residual is 0 and the standard deviation is 1. Standardized residuals are
calculated for each cell in the design. They are useful in helping to interpret chi-square
tables by providing information about which cells contribute to a significant chi-square. 
If the standardized residual is beyond the range of ± 2, then that cell can be considered to
be a major contributor, if it is > +2, or a very weak contributor, if it is beyond -2, to the
overall chi-square value.
<img width="1440" alt="Снимок экрана 2021-09-04 в 16 18 12" src="https://user-images.githubusercontent.com/7578492/132095866-b01c9e77-2ea3-463c-9602-319c2700dabe.png">




