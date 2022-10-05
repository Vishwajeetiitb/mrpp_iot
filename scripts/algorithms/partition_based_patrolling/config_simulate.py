#! /usr/bin/env python3

import configparser as CP
import os
import rospkg
import glob
import rospy
import os
import sys
import networkx as nx
from datetime import datetime

start_time  = datetime.now()

no_of_bots = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
algos = ['iot_communication_network','cr']
graphs = ['iit_bombay','iit_delhi','iit_madras']
iot_device_ranges = [150,250,350,500,10000]
dir_name = rospkg.RosPack().get_path('mrpp_sumo')
no_of_runs = 1

# rospy.init_node('config_simulate')
for graph_name in graphs:
    graph_path = dir_name +'/graph_ml/'+ graph_name + '.graphml'
    graph_net = nx.read_graphml(graph_path)
    for run_id in range(no_of_runs):
        for algo_name in algos:
            how_many_iterations = 1
            if 'iot' in algo_name:
                how_many_iterations = len(iot_device_ranges)
            for idx in range(how_many_iterations):
                for no_agents in no_of_bots:
                    run_start_time = datetime.now()
                    init_locations = " ".join(list(graph_net.nodes())[0:no_agents])
                    os.system("xterm -e roscore & sleep 3")
                    rospy.set_param('/init_locations',init_locations)
                    rospy.set_param('/use_sim_time',True)
                    rospy.set_param('/gui',False)
                    rospy.set_param('/graph',graph_name)
                    rospy.set_param('/init_bots',no_agents)
                    rospy.set_param('done',False)
                    rospy.set_param('/sim_length',30000)
                    rospy.set_param('/algo_name',algo_name)
                    rospy.set_param('/no_of_deads',0)
                    rospy.set_param('/run_id',run_id)
                    rospy.set_param('/random_string','test')
                    if 'iot' in algo_name: rospy.set_param('/iot_device_range',iot_device_ranges[idx])

                    for name in rospy.get_param_names()[4:]:
                        print(name,':',rospy.get_param(name))

                    os.system("xterm -e rosrun mrpp_sumo sumo_wrapper.py & sleep 3")
                    os.system("xterm -e rosrun mrpp_sumo "+ algo_name +".py & sleep 3")
                    os.system("xterm -e rosrun mrpp_sumo command_center.py")
                    os.system("sleep 10")
                    os.system("killall xterm & sleep 3")

                    run_end_time = datetime.now()
                    print('Algorithm took', run_end_time-run_start_time)
                    print('─' * 100,'\n')




end_time  = datetime.now()
print('\n','StartTime:',start_time, '|EndTime:',end_time,'|Total Time taken:',end_time-start_time)

