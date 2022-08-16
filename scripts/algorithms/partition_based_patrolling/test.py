import plotly.express as px
import numpy as np
# Creating the Figure instance
k  = 400
data_x = np.arange(k,1000,0.1)
data_y = (k**2)/(data_x**2)
fig = px.line(x=data_x, y=data_y)
fig.show()

