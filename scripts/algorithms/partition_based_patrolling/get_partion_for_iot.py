#!/usr/bin/env python3

import os
import pickle
import shutil
import sys
from math import *
import pickle5 as pickle

import alphashape
import networkx as nx
import numpy as np
import pandas as pd
import rospkg
from scipy.spatial import ConvexHull, Voronoi, convex_hull_plot_2d
from shapely.geometry import LineString as Shapely_line
from shapely.geometry import Point as Shapely_point
from shapely.geometry import Polygon as Shapely_polygon
from sympy import *


def get_boundary_hull(points):
    global hull, hull_points, hull_poly
    hull_path = dirname+'/graph_ml/'+graph_name+'_hull'
    if os.path.exists(hull_path):
        with open(hull_path, "rb") as poly_file:
            hull = pickle.load(poly_file)
    else:    

        # hull = ConvexHull(initial_points)
        hull =alphashape.alphashape(points)
    
        with open(hull_path, "wb") as poly_file:
            pickle.dump(hull, poly_file, pickle.HIGHEST_PROTOCOL)

    hull_points=np.column_stack((hull.exterior.coords.xy)).tolist()
    hull_points = np.array(hull_points)
    hull_poly = Shapely_polygon(hull_points.tolist()) # Define Convex Hull Polygon


if __name__ == '__main__':
    dirname = rospkg.RosPack().get_path('mrpp_sumo')
    Iot_device_ranges = sorted([10000],reverse=True)

    graph_name = 'iit_bombay'
    graph_path = dirname +'/graph_ml/'+ graph_name + '.graphml'
    graph = nx.read_graphml(graph_path)
    graph_points = []
    for node,data in graph.nodes(data=True):
        graph_points.append(np.array((data['x'],data['y'])))
    get_boundary_hull(graph_points)

    graph_all_results_path = dirname +'/scripts/algorithms/partition_based_patrolling/graphs_partition_results/'+ graph_name + '/'
    if not os.path.exists(graph_all_results_path):
        os.makedirs(graph_all_results_path)
        
    no_of_base_stations = 1
    rho_max = None

    for range in Iot_device_ranges:
        
        while True:
            print('python3 ' + dirname +'/scripts/algorithms/partition_based_patrolling/graph_partition.py '+ graph_name + ' ' + str(no_of_base_stations))
            os.system('python3 ' + dirname +'/scripts/algorithms/partition_based_patrolling/graph_partition.py '+ graph_name + ' ' + str(no_of_base_stations))
            graph_results_path = dirname +'/scripts/algorithms/partition_based_patrolling/graphs_partition_results/'+ graph_name + '/' + str(no_of_base_stations) + '_base_stations/'
            base_stations_df   = pd.read_csv(graph_results_path + graph_name + "_with_"+str(no_of_base_stations) + '_base_stations.csv',converters={'location': pd.eval,'Radius': pd.eval})
            rho_max = max(base_stations_df['Radius'])
            if rho_max is not None and rho_max < range:
                base_stations_df.to_csv(dirname +'/scripts/algorithms/partition_based_patrolling/graphs_partition_results/'+ graph_name + '/' + str(range) + '_range_base_stations.csv')
                break
            no_of_base_stations +=1
