#! /usr/bin/env python3

import configparser as CP
import os
import rospkg
import glob
import rospy
import os
import sys
import networkx as nx


no_of_bots = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
algos = ['iot_communication_network','cr']
graphs = ['iitb_full']
iot_device_ranges = [150,250,350,500,10000]
how_many_iterations = 1 # This variable is to iterate over ranges if algo is based on IoT
dir_name = rospkg.RosPack().get_path('mrpp_sumo')

# rospy.init_node('config_simulate')
for graph_name in graphs:
    graph_path = dir_name +'/graph_ml/'+ graph_name + '.graphml'
    graph_net = nx.read_graphml(graph_path)
    
    for algo_name in algos:
        for no_agents in no_of_bots:
            if 'iot' in algo_name:
                how_many_iterations = len(iot_device_ranges)
            else : how_many_iterations = 1
            for idx in range(how_many_iterations):
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
                rospy.set_param('/run_id',0)
                # rospy.set_param('/random_string','test')

                if 'iot' in algo_name: 
                    rospy.set_param('/random_string',algo_name+'_'+str(iot_device_ranges[idx])+'_'+str(no_agents)+'agents')
                    os.system("xterm -e rosrun mrpp_sumo sumo_wrapper.py & sleep 3")
                    os.system("xterm -e rosrun mrpp_sumo "+ algo_name +".py " + str(iot_device_ranges[idx]) + " & sleep 3")
                else :
                    rospy.set_param('/random_string',algo_name+'_'+str(no_agents)+'agents')
                    os.system("xterm -e rosrun mrpp_sumo sumo_wrapper.py & sleep 3")
                    os.system("xterm -e rosrun mrpp_sumo "+ algo_name +".py & sleep 3")

                os.system("xterm -e rosrun mrpp_sumo command_center.py")
                os.system("sleep 10")
                os.system("killall xterm & sleep 3")


