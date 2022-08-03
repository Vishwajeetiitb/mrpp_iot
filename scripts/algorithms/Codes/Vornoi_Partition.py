import string
from turtle import color, right
from matplotlib import style
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
import random
from sympy import *
from scipy._lib.decorator import decorator as _decorator
import smallestenclosingcircle 
from scipy.stats import norm
import statistics
import datetime


__all__ = ['delaunay_plot_2d', 'convex_hull_plot_2d', 'voronoi_plot_2d']


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
    ptp_bound = vor.points.ptp(axis=0)

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


                    
# # Intersection of Convex hull with finite and Infinite  segments of voronoi Diagram 
#     vor_boundary_points = []
#     for segment in finite_segments:
#       line = Polygon(Point(segment[0]),Point(segment[1]))
#       intersection = hull_poly.intersection(line)
#       if intersection != []:
#         vor_boundary_points.append(intersection)

#     for segment in Infinite_segments:
#       line = Polygon(Point(segment[0]),Point(segment[1]))
#       intersection = hull_poly.intersection(line)
#       if intersection != []:
#         vor_boundary_points.append(intersection)
    
#     vor_boundary_points = np.array(vor_boundary_points)[:,0,:]



# Clip the vornoi Regions inside Convex hull
    final_regions = []
    for point in vor.points:
        if vor_hull_poly.encloses_point(point):
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
            # Finding segments of hull which cut the boundary regions

            enclosed_point_index = np.where(vor.points==point)[0][0] 
            Infinite_region_index = vor.point_region[enclosed_point_index]
            Infinite_region_points_indices = vor.regions[Infinite_region_index]
            Infinite_region_points_indices.remove(-1)
            Infinite_region = vor.vertices[Infinite_region_points_indices]
            Infinite_region_temp = Infinite_region
            for p in Infinite_region:
                if hull_poly.encloses_point(p) == False:
                    Infinite_region_temp = np.delete(Infinite_region_temp,np.where(Infinite_region_temp==p)[0][0],0)
            Infinite_region = Infinite_region_temp



            # Deriving segments of region of point

            point_index = np.where(vor.points==point)[0][0]
            point_ridge_indices =np.where(vor.ridge_points==point_index)[0]
            ridge_segment_vertices = np.take(vor.ridge_vertices,point_ridge_indices,axis=0)
            region_segment_index = None
            region_segment = None

            for ridge_segment_vertex in ridge_segment_vertices:
                
                if -1 in ridge_segment_vertex:
                    ridge_point = vor.vertices[ridge_segment_vertex[1]]
                    region_segment_index = np.where(Infinite_segments==ridge_point)[0][0]
                    region_segment = np.take(Infinite_segments,region_segment_index,axis=0)
                else:

                    ridge_points = vor.vertices[ridge_segment_vertex]
                    region_segment_index = np.where((finite_segments==ridge_points).all(axis=1))[0][0]
                    region_segment = np.take(finite_segments,region_segment_index,axis=0)
                    
                region_segment = Segment(*tuple(region_segment))
                hull_intersection = region_segment.intersection(hull_poly)
                if hull_intersection !=[]:
                    Infinite_region = np.append(Infinite_region,hull_intersection,axis=0)

                

            Infinite_region = np.append(Infinite_region,[point],axis=0)
            final_regions.append(Infinite_region)  #appending to final set of polygon and its seed

    # ax.plot(vor_boundary_points[:,0],vor_boundary_points[:,1],'*',markersize=7)

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
    for simplex in hull.simplices:
      ax.plot(initial_points[simplex, 0], initial_points[simplex, 1], 'k-')
    _adjust_bounds(ax, hull.points)
    return ax,np.array(final_regions,dtype=object)



rng = np.random.default_rng(12345)
old_points = rng.random((25, 2))
initial_points = old_points
hull = ConvexHull(initial_points)
hull_points = []
for i in hull.vertices:
    hull_points.append(Point(initial_points[i]))

hull_points = np.array(hull_points)
hull_poly = Polygon(*tuple(hull_points)) # Define Convex Hull Polygon

for i in range(100):
    a = datetime.datetime.now()
    radii = []
    voronoi = Voronoi(old_points)
    plt_ax,cliped_regions = voronoi_plot_2d_clip(voronoi)
    new_points = []
    for region in cliped_regions:
        c_x,c_y,r = smallestenclosingcircle.make_circle(region)
        new_points.append([c_x,c_y])
        radii.append(r)
        enclosing_circle = plt.Circle(( c_x , c_y ), r ,fill=False,color='#34eb43')
        plt_ax.add_artist(enclosing_circle)
    old_points = new_points
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.clf()
    x_axis = np.array(radii)
    x_axis = np.sort(x_axis)
    mean = statistics.mean(x_axis)
    sd = statistics.stdev(x_axis)
    b= datetime.datetime.now()
    c= b-a

    print(mean,sd,len(new_points),c.seconds)
    # plt.plot(x_axis, norm.pdf(x_axis, mean, sd))
    # plt.show()
    plt.savefig('./partition_stage_results/partition_stage'+str(i))
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

