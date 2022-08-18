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
no_agents_list = [1,3,5,7,9]
algo_list = ['mrpp_iot_250','mrpp_iot_350','mrpp_iot_500']
steady_time_stamp = 3000
dirname = rospkg.RosPack().get_path('mrpp_sumo')
available_comparisons = ['avg_idleness', 'worst_idleness']
comparison_parameter_index = 0


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
        idle = np.load(dirname+ "/post_process/"  + graph_name+ "/"+ algo_name + "/" + str(no_agents)+ "_agents/data.npy")
        stamps = np.load(dirname+ "/post_process/" + graph_name+ "/"+ algo_name + "/"  + str(no_agents)+ "_agents/stamps.npy")
        idle = idle[np.argwhere(stamps>steady_time_stamp)[0][0]:]  # Taking idlness values after steady state
        worst_idles = idle.max(axis=0)
        avg_idles = np.average(idle,axis=0)
        for avg_node_idle,worst_node_idle in zip(avg_idles,worst_idles):
            idle_dic = {'Average Idleness' : [avg_node_idle],'Worst Idleness' : [worst_node_idle],'Algorithm': [algo_name], 'Agents' : [no_agents]}
            df = pd.concat([df,pd.DataFrame(idle_dic,index=[0])])

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
    print("http://vishwajeetiitb.github.io/mrpp_iot//scripts/algorithms/partition_based_patrolling/plot/"+ graph_name + '/avg_node_idle/' + urllib.parse.quote(file_name))

elif comparison_parameter_index ==1:
    fig = px.box(df, x="Agents", y="Worst Idleness", color="Algorithm")
    plot_dir = dirname + '/scripts/algorithms/partition_based_patrolling/plot/'+ graph_name + '/wrost_node_idle/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    fig.write_html(plot_dir+file_name)
    print("http://vishwajeetiitb.github.io/mrpp_iot//scripts/algorithms/partition_based_patrolling/plot/"+ graph_name + '/wrost_node_idle/' + urllib.parse.quote(file_name))

# fig.show()