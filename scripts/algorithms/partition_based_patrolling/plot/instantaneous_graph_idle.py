from turtle import color, stamp, title
from unicodedata import name
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import rospkg
import pandas as pd
import os
from slugify import slugify
import urllib.parse

dirname = rospkg.RosPack().get_path('mrpp_sumo')
no_agents_list = [1,3,5,7]
algo_list = ['mrpp_iot_500','mrpp_iot2_500','mrpp_iot3_500','cr']
row_size = 2
col_size = 2
graph_name = 'iitb_full'
algo_name = 'mrpp_iot2_250'
color_list = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]

fig = make_subplots(rows=row_size, cols=col_size,subplot_titles=[str(i)+ " Agents" for i in no_agents_list])


for idx,no_agents in enumerate(no_agents_list):
    
    for m,algo_name in enumerate(algo_list):
        df = pd.DataFrame()
        idle = np.load(dirname+ "/post_process/"  + graph_name+ "/"+ algo_name + "/" + str(no_agents)+ "_agents/data.npy")
        stamps = np.load(dirname+ "/post_process/" + graph_name+ "/"+ algo_name + "/"  + str(no_agents)+ "_agents/stamps.npy")
        graph_idle = np.average(idle,axis=1)
        fig.add_trace(go.Scatter(x=stamps, y=graph_idle,mode='lines',marker=dict(color=color_list[m]),legendgroup=m+1,name=algo_name,showlegend=(True if idx==0 else False)),row=int(idx/row_size)+1,col=idx%row_size+1)
    

    fig['layout']['xaxis'+str(idx+1)]['title']='Stamps'
    fig['layout']['yaxis'+str(idx+1)]['title']='Instantaneous Graph Idleness'

fig.update_layout(title='Instantaneous Graph Idleness Plot',title_x=0.5)
# fig.show()


file_name = ""
for idx,algo in enumerate(algo_list):
    if not idx:
        file_name = algo
    else:
        file_name = file_name + " | " + algo
        
file_name = file_name + ".html"
plot_dir = dirname + '/scripts/algorithms/partition_based_patrolling/plot/'+ graph_name + '/instantaneous_graph_idle/'

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

fig.write_html(plot_dir+file_name)

print("http://vishwajeetiitb.github.io/mrpp_iot//scripts/algorithms/partition_based_patrolling/plot/"+ graph_name + '/instantaneous_graph_idle/' + urllib.parse.quote(file_name))



