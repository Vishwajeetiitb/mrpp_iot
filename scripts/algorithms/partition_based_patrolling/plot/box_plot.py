from turtle import color
import plotly.express as px
import plotly.graph_objects as go
import rospkg
import numpy as np
import pandas as pd
import os 
from slugify import slugify
import urllib.parse


graph_name = 'iitb_full'
no_agents_list = [3,5,7,9,11,13]
algo_list = ['iot_communication_network_150','iot_communication_network_250','iot_communication_network_350','iot_communication_network_10000','cr']
steady_time_stamp = 3000
dirname = rospkg.RosPack().get_path('mrpp_sumo')
available_comparisons = ['avg_idleness', 'worst_idleness']


stamp_as_points = False
comparison_parameter_index = 1

# fig = go.Figure(layout=go.Layout(
#                 title=graph_name +' Average node Idleness distribution',
#                 titlefont_size=16,
#                 showlegend=False,
#                 hovermode='closest',
#                 margin=dict(b=20,l=5,r=5,t=40),
#                 # annotations=[ dict(
#                 #     text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
#                 #     showarrow=False,
#                 #     xref="paper", yref="paper",
#                 #     x=0.005, y=-0.002 ) ],
#                 yaxis=dict(scaleanchor="x", scaleratio=1)))

df = pd.DataFrame()
for no_agents in no_agents_list:
    for algo_name in algo_list:  
        idle = np.load(dirname+ "/post_process/"  + graph_name+ "/"+ algo_name + "/" + str(no_agents)+ "_agents/data_final.npy")
        stamps = np.load(dirname+ "/post_process/" + graph_name+ "/"+ algo_name + "/"  + str(no_agents)+ "_agents/stamps_final.npy")
        idle = idle[np.argwhere(stamps>steady_time_stamp)[0][0]:]  # Taking idlness values after steady state
        worst_idles = idle.max(axis=int(stamp_as_points))
        avg_idles = np.average(idle,axis=int(stamp_as_points))
        df_temp = pd.DataFrame()

        df_temp['Worst Idleness'] = worst_idles
        df_temp['Average Idleness'] = avg_idles
        df_temp['Algorithm']= [algo_name]*idle.shape[int(not stamp_as_points)]
        df_temp['Agents'] = [no_agents]*idle.shape[int(not stamp_as_points)]
        df = pd.concat([df,df_temp])


# fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default



file_name = ""
for idx,algo in enumerate(algo_list):
    if not idx:
        file_name = algo
    else:
        file_name = file_name + " | " + algo
        
file_name = file_name + ".html"


if comparison_parameter_index ==0:
    fig = px.box(df, x="Agents", y="Average Idleness", color="Algorithm")
    plot_dir = dirname + '/scripts/algorithms/partition_based_patrolling/plot/'+ graph_name + '/avg_node_idle/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    fig.write_html(plot_dir+file_name)
    print("http://vishwajeetiitb.github.io/mrpp_iot/scripts/algorithms/partition_based_patrolling/plot/"+ graph_name + '/avg_node_idle/' + urllib.parse.quote(file_name))

elif comparison_parameter_index ==1:
    fig = px.box(df, x="Agents", y="Worst Idleness", color="Algorithm")
    plot_dir = dirname + '/scripts/algorithms/partition_based_patrolling/plot/'+ graph_name + '/wrost_node_idle/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    fig.write_html(plot_dir+file_name)
    print("http://vishwajeetiitb.github.io/mrpp_iot/scripts/algorithms/partition_based_patrolling/plot/"+ graph_name + '/wrost_node_idle/' + urllib.parse.quote(file_name))

fig.show()