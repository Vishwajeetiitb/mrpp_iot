from matplotlib import axis
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
from sympy import *
import networkx as nx
from scipy._lib.decorator import decorator as _decorator
from yaml import Mark
import smallestenclosingcircle 
import statistics
import datetime
import xml.etree.ElementTree as ET
from math import *
import os
import shutil
from shapely.geometry import Polygon as Shapely_polygon
from shapely.geometry import LineString as Shapely_line
from shapely.geometry import Point as Shapely_point
from shapely.ops import split as Shapely_split
from shapely.ops import cascaded_union as Shapely_cascaded_union
import sys




graph_name = 'cair'
no_of_base_stations = 4
graph_path = './graphs/'+ graph_name + '.graphml'
graph_results_path = './graphs_partition_results/'+ graph_name + '/' + str(no_of_base_stations) + '_base_stations/'
graph = nx.read_graphml(graph_path)
if os.path.exists(graph_results_path):
    shutil.rmtree(graph_results_path)
    os.makedirs(graph_results_path)
else:
    os.makedirs(graph_results_path)


__all__ = ['delaunay_plot_2d', 'convex_hull_plot_2d', 'voronoi_plot_2d_clip']


@_decorator
def _held_figure(func, obj, ax=None, **kw):
    import matplotlib.pyplot as plt  # type: ignore[import]

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
        return func(obj, ax=ax, **kw)

    # As of matplotlib 2.0, the "hold" mechanism is deprecated.
    # When matplotlib 1.x is no longer supported, this check can be removed.
    was_held = getattr(ax, 'ishold', lambda: True)()
    if was_held:
        return func(obj, ax=ax, **kw)
    try:
        ax.hold(True)
        return func(obj, ax=ax, **kw)
    finally:
        ax.hold(was_held)


def _adjust_bounds(ax, points):
    margin = 0.1 * points.ptp(axis=0)
    xy_min = points.min(axis=0) - margin
    xy_max = points.max(axis=0) + margin
    ax.set_xlim(xy_min[0], xy_max[0])
    ax.set_ylim(xy_min[1], xy_max[1])


@_held_figure
def delaunay_plot_2d(tri, ax=None):

    if tri.points.shape[1] != 2:
        raise ValueError("Delaunay triangulation is not 2-D")

    x, y = tri.points.T
    ax.plot(x, y, 'o')
    ax.triplot(x, y, tri.simplices.copy())

    _adjust_bounds(ax, tri.points)

    return ax.figure


@_held_figure
def convex_hull_plot_2d(hull, ax=None):

    from matplotlib.collections import LineCollection  # type: ignore[import]

    if hull.points.shape[1] != 2:
        raise ValueError("Convex hull is not 2-D")

    ax.plot(hull.points[:,0], hull.points[:,1], 'o')
    line_segments = [hull.points[simplex] for simplex in hull.simplices]
    ax.add_collection(LineCollection(line_segments,
                                     colors='k',
                                     linestyle='solid'))
    _adjust_bounds(ax, hull.points)

    return ax.figure


@_held_figure
def voronoi_plot_2d_clip(vor, ax=None, **kw):

    from matplotlib.collections import LineCollection

    if vor.points.shape[1] != 2:
        raise ValueError("voronoi diagram is not 2-D")
    
    if kw.get('show_points', True):
        point_size = kw.get('point_size', None)
        ax.plot(vor.points[:,0], vor.points[:,1], '.', markersize=point_size)
    if kw.get('show_vertices', True):
        ax.plot(vor.vertices[:,0], vor.vertices[:,1], 'o',markersize=1)

    line_colors = kw.get('line_colors', 'k')
    line_width = kw.get('line_width', 1.0)
    line_alpha = kw.get('line_alpha', 1.0)

    center = vor.points.mean(axis=0)
    ptp_bound = hull_points.ptp(axis=0)

    finite_segments = []
    Infinite_segments = []

    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            finite_segments.append(vor.vertices[simplex])
        else:
            i = simplex[simplex >= 0][0]  # finite end voronoi vertex

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            if (vor.furthest_site):
                direction = -direction
            far_point = vor.vertices[i] + direction * ptp_bound.max()

            Infinite_segments.append([vor.vertices[i], far_point])
    vor_hull = ConvexHull(vor.points)
    vor_hull_points = []
    for i in vor_hull.vertices:
        vor_hull_points.append(Point(vor.points[i]))

    vor_hull_points = np.array(vor_hull_points)
    vor_hull_poly = Polygon(*tuple(vor_hull_points)) # Define Convex Hull Polygon



# Clip the vornoi Regions inside Convex hull
    final_regions = []
    test = 0
    for point in vor.points:
        enclosed_point_index = np.where(vor.points==point)[0][0] 
        region_index = vor.point_region[enclosed_point_index]
        region_points_indices = vor.regions[region_index]
        
        if -1 not in region_points_indices:
            enclosed_point_index = np.where(vor.points==point)[0][0] 
            finite_region_index = vor.point_region[enclosed_point_index]
            finite_region_points_indices = vor.regions[finite_region_index]
            finite_region = vor.vertices[finite_region_points_indices]
            finite_region_poly = Polygon(*tuple(finite_region))
            clip_region_points = hull_poly.intersection(finite_region_poly)
            finite_region_temp = finite_region
            for p in finite_region:
                if hull_poly.encloses_point(p) == False:
                    finite_region_temp = np.delete(finite_region_temp,np.where(finite_region_temp==p)[0][0],0)
            finite_region = finite_region_temp
            for clip in clip_region_points:
                finite_region = np.append(finite_region,[clip],axis=0)
            final_regions.append(finite_region)  #appending to final set of polygon and its seed

        else:
            # Finding regions of hull which cut the boundary regions

            enclosed_point_index = enclosed_point_index
            Infinite_region_index = region_index
            Infinite_region_points_indices = region_points_indices

            while Infinite_region_points_indices.index(-1) !=0:
                Infinite_region_points_indices.append(Infinite_region_points_indices.pop(0))
            Infinite_region_points_indices.remove(-1)
            
            Infinite_region = vor.vertices[Infinite_region_points_indices]
            point_index = np.where(vor.points==point)[0][0]
            point_ridge_indices =np.where(vor.ridge_points==point_index)[0]
            ridge_segment_vertices = np.take(vor.ridge_vertices,point_ridge_indices,axis=0)

            for ridge_segment_vertex, m in zip(ridge_segment_vertices,range(ridge_segment_vertices.shape[0])):
            
                if -1 in ridge_segment_vertex:
                    pointidx = vor.ridge_points[point_ridge_indices[m]]
                    i = ridge_segment_vertex[ridge_segment_vertex >= 0][0]  # finite end voronoi vertex

                    t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
                    t /= np.linalg.norm(t)
                    n = np.array([-t[1], t[0]])  # normal

                    midpoint = vor.points[pointidx].mean(axis=0)
                    direction = np.sign(np.dot(midpoint - center, n)) * n
                    if (vor.furthest_site):
                        direction = -direction
                    far_point = vor.vertices[i] + direction * ptp_bound.max()
                    if np.where(Infinite_region==vor.vertices[i])[0][0] == 0:
                        Infinite_region = np.insert(Infinite_region,0,far_point,axis=0)
                    else:
                        Infinite_region = np.append(Infinite_region,[far_point],axis=0)
                  
            a = Shapely_polygon(hull_points.tolist())
            b = Shapely_line(Infinite_region.tolist())
            x,y = b.coords.xy
            color_list = ['r','g','c','b','m','k']
            # ax.plot(x,y,'-k',color=color_list[test],linewidth=5-test)
            # test+=1
            c = Shapely_split(a,b) 
            issue = True
            for poly in c.geoms:
                if poly.contains(Shapely_point(point)) or poly.intersects(Shapely_point(point)):
                    x,y = poly.exterior.coords.xy
                    Infinite_region = np.column_stack((x,y))
                    issue = False
            if issue: 
                print('Issue detected')
                for idx,d in enumerate(c.geoms):
                    x,y = d.exterior.coords.xy
                    ax.plot(x,y,'-k',color=color_list[idx+1],linewidth=4-idx)
                x,y = b.coords.xy
                ax.plot(x,y,'-k',color=color_list[0],linewidth=1)
                plt.show()
                sys.exit()
            final_regions.append(Infinite_region)  #appending to final set of polygon and its seed


    ax.add_collection(LineCollection(finite_segments,
                                     colors=line_colors,
                                     lw=line_width,
                                     alpha=line_alpha,
                                     linestyle='solid'))
    ax.add_collection(LineCollection(Infinite_segments,
                                     colors=line_colors,
                                     lw=line_width,
                                     alpha=line_alpha,
                                     linestyle='dashed'))

    ax.plot(vor.points[:,0], vor.points[:,1], 'o',markersize=1.5)
    hull_x,hull_y = zip(*np.append(hull_points,[hull_points[0]],axis=0))
    ax.plot(hull_x, hull_y, 'k-')
    _adjust_bounds(ax, hull.points)
    return ax,np.array(final_regions,dtype=object)
def base_station_initial_points(boundary_poly,n):

    minx, miny, maxx, maxy = boundary_poly.bounds
    random_x = None
    random_y = None
    base_station_points = []
    for i in range(n):
        is_inside = False
        while not is_inside:
            random_x = np.random.uniform( minx, maxx, 1 )[0]
            random_y = np.random.uniform( miny, maxy, 1 )[0]
            is_inside = boundary_poly.encloses_point([random_x,random_y])
        base_station_points.append([random_x,random_y])
    return base_station_points

rng = np.random.default_rng(12345)
# old_points = rng.random((25, 2))
old_points = []

for node,data in graph.nodes(data=True):
    old_points.append(np.array((data['x'],data['y'])))

initial_points = old_points
hull = ConvexHull(initial_points)
hull_points = []
for i in hull.vertices:
    hull_points.append(Point(initial_points[i]))

hull_points = np.array(hull_points)
hull_poly = Polygon(*tuple(hull_points)) # Define Convex Hull Polygon
starting_base_points = base_station_initial_points(hull_poly,n=no_of_base_stations)
voronoi = Voronoi(starting_base_points)
new_base_points = starting_base_points
sd = 1
i = 0
while sd > 0.002:
    a = datetime.datetime.now()
    voronoi = Voronoi(new_base_points)
    previous_base_points = new_base_points
    plt_ax,cliped_regions = voronoi_plot_2d_clip(voronoi)
    radii = []
    new_base_points = []

    for idx,region in enumerate(cliped_regions):
        c_x,c_y,r = smallestenclosingcircle.make_circle(region)
        region_poly = Polygon(*tuple(region))
        if region_poly.encloses_point(Point([c_x,c_y])):
            new_base_points.append([c_x,c_y])
        else:
            # p = previous_base_points[idx]
            p = base_station_initial_points(region_poly,1)[0]
            new_base_points.append(p)

        radii.append(r)
        enclosing_circle = plt.Circle(( c_x , c_y ), r ,fill=False,color='#34eb43')
        plt_ax.add_artist(enclosing_circle)
    
    x_axis = np.array(radii)
    x_axis = np.sort(x_axis)
    mean = statistics.mean(x_axis)
    sd = statistics.stdev(x_axis)
    b= datetime.datetime.now()
    c= b-a
    print(i,mean,sd,len(starting_base_points),c.seconds)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(graph_results_path+'partition_stage'+str(i)+'.png')
    # plt.show()
    plt.close('all')
    i+=1
np.save(graph_results_path + graph_name + "_with_"+str(no_of_base_stations) + '_base_stations.npy',new_base_points)