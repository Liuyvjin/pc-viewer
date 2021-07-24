""" Utility functions for processing point clouds.

Author: Charles R. Qi, Hao Su
Date: November 2016
Modified by: Liu Jin
Modified date: 2021/7/20
"""

import os
import sys
from functools import reduce

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Point cloud IO
import numpy as np
from plyfile import PlyData, PlyElement


# ----------------------------------------
# Point Cloud/Volume Conversions
# ----------------------------------------

def point_cloud_to_volume_batch(point_clouds, vsize=12, radius=1.0, flatten=True):
    """ Input is BxNx3 batch of point cloud
        Output is Bx(vsize^3)
    """
    vol_list = []
    for b in range(point_clouds.shape[0]):
        vol = point_cloud_to_volume(np.squeeze(point_clouds[b,:,:]), vsize, radius)
        if flatten:
            vol_list.append(vol.flatten())
        else:
            vol_list.append(np.expand_dims(np.expand_dims(vol, -1), 0))
    if flatten:
        return np.vstack(vol_list)
    else:
        return np.concatenate(vol_list, 0)


def point_cloud_to_volume(points, vsize, radius=1.0):
    """ input is Nx3 points.
        output is vsize*vsize*vsize
        assumes points are in range [-radius, radius]
    """
    vol = np.zeros((vsize,vsize,vsize))
    voxel = 2*radius/float(vsize)
    locations = (points + radius)/voxel
    locations = locations.astype(int)
    vol[locations[:,0],locations[:,1],locations[:,2]] = 1.0
    return vol

#a = np.zeros((16,1024,3))
#print point_cloud_to_volume_batch(a, 12, 1.0, False).shape

def volume_to_point_cloud(vol):
    """ vol is occupancy grid (value = 0 or 1) of size vsize*vsize*vsize
        return Nx3 numpy array.
    """
    vsize = vol.shape[0]
    assert(vol.shape[1] == vsize and vol.shape[1] == vsize)
    points = []
    for a in range(vsize):
        for b in range(vsize):
            for c in range(vsize):
                if vol[a,b,c] == 1:
                    points.append(np.array([a,b,c]))
    if len(points) == 0:
        return np.zeros((0,3))
    points = np.vstack(points)
    return points

# ----------------------------------------
# Point cloud IO
# ----------------------------------------

def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x,y,z in pc])
    return pc_array


def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)


# ----------------------------------------
# Simple Point cloud and Volume Renderers
# ----------------------------------------

def draw_point_cloud(input_points, canvasSize=500, space=200, diameter=25,
                     xrot=0, yrot=0, zrot=0, switch_xyz=[0,1,2], normalize=True):
    """ Render point cloud to image with alpha channel.
        Input:
            points: Nx3 numpy array (+y is up direction)
        Output:
            gray image as numpy array of size canvasSize x canvasSize
    """
    image = np.zeros((canvasSize, canvasSize))
    if input_points is None or input_points.shape[0] == 0:
        return image

    points = input_points[:, switch_xyz]
    M = euler2mat(zrot, yrot, xrot)
    points = (np.dot(M, points.transpose())).transpose()

    # Normalize the point cloud
    # We normalize scale to fit points in a unit sphere
    if normalize:
        centroid = np.mean(points, axis=0)
        points -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
        points /= furthest_distance

    # Pre-compute the Gaussian disk
    radius = (diameter-1)/2.0
    disk = np.zeros((diameter, diameter))
    for i in range(diameter):
        for j in range(diameter):
            if (i-radius) * (i-radius) + (j-radius) * (j-radius) <= radius * radius:
                disk[i, j] = np.exp((-(i-radius)**2 - (j-radius)**2)/(radius**2))
    dx, dy= (disk > 0).nonzero()
    dv = disk[disk > 0]

    # Order points by z-buffer
    zorder = np.argsort(points[:, 2])
    points = points[zorder, :]
    points[:, 2] = (points[:, 2] - np.min(points[:, 2])) / (np.max(points[:, 2] - np.min(points[:, 2])))
    max_depth = 1 #np.max(points[:, 2])


    for i in range(points.shape[0]):
        j = points.shape[0] - i - 1
        x = points[j, 0]
        y = points[j, 1]
        xc = canvasSize/2 + (x*space)
        yc = canvasSize/2 + (y*space)
        xc = int(np.round(xc))
        yc = int(np.round(yc))

        px = dx + xc
        py = dy + yc

        image[px, py] = image[px, py] * 0.5 + dv * (max_depth - points[j, 2]) * 0.5

    image = image / np.max(image)
    return image

def point_cloud_three_views(points, diameter=10):
    """ input points Nx3 numpy array (+y is up direction).
        return an numpy array gray image of size 500x1500. """
    # +y is up direction
    # xrot is azimuth
    # yrot is in-plane
    # zrot is elevation
    img1 = draw_point_cloud(points, zrot=110/180.0*np.pi, xrot=45/180.0*np.pi, yrot=0/180.0*np.pi, diameter=diameter)
    img2 = draw_point_cloud(points, zrot=70/180.0*np.pi, xrot=135/180.0*np.pi, yrot=0/180.0*np.pi, diameter=diameter)
    img3 = draw_point_cloud(points, zrot=180.0/180.0*np.pi, xrot=90/180.0*np.pi, yrot=0/180.0*np.pi, diameter=diameter)
    image_large = np.concatenate([img1, img2, img3], 1)
    return image_large


# ----------------------------------------
# matplotlib Point cloud and Volume Renderers
# ----------------------------------------

import matplotlib.pyplot as plt
def pyplot_draw_point_cloud(points, output_filename=None):
    """ points is a Nx3 numpy array """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    if output_filename is not None:
        plt.savefig(output_filename)

def pyplot_draw_volume(vol, output_filename):
    """ vol is of size vsize*vsize*vsize
        output an image to output_filename
    """
    points = volume_to_point_cloud(vol)
    pyplot_draw_point_cloud(points, output_filename)


# ----------------------------------------
#  Point cloud Render with RGB
# ----------------------------------------

def draw_pointcloud_rgb(    pointcloud,
                            rgb             =   None,
                            alpha           =   0.5,
                            canvasSize      =   500,
                            space           =   200,
                            diameter        =   20,
                            rot             =   [0, 0, 0],
                            switch_xyz      =   [0,1,2],
                            normalize       =   True,
                            depth_decrease  =   0.5,
                            light           =   True,
                            back_color      =   [255,255,255]   ):
    """ Render point cloud to image with alpha channel.
        Input:
            points: Nx3 numpy array (+y is up direction)
        Output:
            image as numpy array of size canvasSize x canvasSize x 3 [or 1]
            if rgb is None, output is a gray image
    """
    # initialize image canvas
    if pointcloud is None or pointcloud.shape[0] == 0:
        raise ValueError
    if rgb is not None:
        rgb = np.array(rgb)
        image = np.ones((canvasSize, canvasSize,3))*back_color
        if len(rgb.shape) == 1:
            rgb = np.tile(rgb, (pointcloud.shape[0], 1))
    else:
        image = np.zeros((canvasSize, canvasSize))

    # rotate the pointcloud, rot can be euler angle list or rotate matrix
    points = pointcloud[:, switch_xyz]
    if isinstance(rot, list):
        rot = euler2mat(*rot)
    points = (np.dot(rot, points.transpose())).transpose()

    # Normalize the point cloud
    if normalize:
        points = normalize_to_unit_sphere(points)

    # Pre-compute the Gaussian disk
    radius = diameter//2
    if light:
        delta = 1/3 * radius
        x_light, y_light = radius-delta, radius+delta
    else:
        x_light, y_light = radius, radius

    alpha_disk = bresenham_circle_alpha_disk(diameter, alpha)
    dx, dy= (alpha_disk > 0).nonzero()
    alpha_disk_val = alpha_disk[dx, dy]
    color_disk_val = np.exp((-(dx-x_light)**2 - (dy-y_light)**2)/(radius**2))
    if rgb is not None:
        alpha_disk_val = alpha_disk_val.reshape(-1,1).repeat(3, axis=1)
        color_disk_val = color_disk_val.reshape(-1,1).repeat(3, axis=1)

    # Order points by z-buffer
    #   from zmin to zmax: depth_factor = 1 ~ depth_decrease
    if points.shape[0] > 1 :
        zorder = np.argsort(points[:, 2])
        points = points[zorder, :]
        zmax, zmin = points[-1, 2], points[0, 2]
        depth_factor = (zmax - points[:, 2]) / (zmax - zmin) * (1-depth_decrease) + depth_decrease
    else:
        depth_factor = [1,]

    # draw points
    for i in range(points.shape[0]):
        j = points.shape[0] - i - 1
        x = points[j, 0]
        y = points[j, 1]
        xc = int(canvasSize/2 + (x*space))
        yc = int(canvasSize/2 + (y*space))

        px = dx + xc
        py = dy + yc
        if (px >= canvasSize).any() or (py>=canvasSize).any() or \
            (px < 0).any() or (py < 0).any():
            continue
        if rgb is not None :
            front_color = color_disk_val * depth_factor[j] * rgb[j]
            image[px, py] = image[px, py]*(1-alpha_disk_val) + front_color * alpha_disk_val
        else:
            front_color = color_disk_val * depth_factor[j]
            image[px, py] = image[px, py]*(1-alpha_disk_val) + front_color * alpha_disk_val

    if rgb is not None:
        image = np.uint8(image)
    else:
        image = np.uint8(image * 255)
    return image



def bresenham_circle_alpha_disk(diameter, alpha):
    radius = (diameter - 1) / 2
    x0, y0 =  radius, radius
    dx, dy = diameter//2-x0, -y0
    delta = 2*(1 + dx + dy)

    alpha_disk = np.zeros((diameter, diameter))
    while dy < 0:
        dist = radius - np.sqrt(dy**2 + dx**2)  # -0.5~0.5
        edge_alpha = dist+0.5
        alpha_disk[int(y0+dy), int(x0+dx)] = edge_alpha
        alpha_disk[int(y0-dx), int(x0+dy)] = edge_alpha
        alpha_disk[int(y0+dx), int(x0-dy)] = edge_alpha
        alpha_disk[int(y0-dy), int(x0-dx)] = edge_alpha

        if (delta>0):
            deltaDV = 2*(delta-dx)-1
            direction = 2 if deltaDV<=0 else 1
        elif (delta<0):
            deltaHD = 2*(delta-dy)-1
            direction = 3 if deltaHD<=0 else 2
        elif (delta == 0):
            direction = 2

        if direction == 1:
            dy += 1
            delta = delta + 2*dy + 1
        elif direction ==2:
            dx += 1
            dy += 1
            delta = delta + 2*(dx+dy+1)
        elif direction ==3:
            dx += 1
            delta = delta + 2*dx + 1
    # fill with 1
    for i in range(1, diameter-1):
        j = 0
        while alpha_disk[i, j]==0:
            j+=1
        while alpha_disk[i, j]:
            j+=1
        while alpha_disk[i, j]==0:
            alpha_disk[i, j]=1
            j+=1

    return alpha_disk



# ----------------------------------------
# Tool function
# ----------------------------------------
def normalize_to_unit_sphere(points):
    """ normalize scale to fit points in a unit sphere """
    centroid = np.mean(points, axis=0)
    points -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
    if furthest_distance != 0:
        points /= furthest_distance
    return points


def euler2mat(z=0, y=0, x=0):
    """ copy from eulerangles """
    Ms = []
    if z:
        cosz = np.cos(z)
        sinz = np.sin(z)
        Ms.append(np.array(
                [[cosz, -sinz, 0],
                 [sinz, cosz, 0],
                 [0, 0, 1]]))
    if y:
        cosy = np.cos(y)
        siny = np.sin(y)
        Ms.append(np.array(
                [[cosy, 0, siny],
                 [0, 1, 0],
                 [-siny, 0, cosy]]))
    if x:
        cosx = np.cos(x)
        sinx = np.sin(x)
        Ms.append(np.array(
                [[1, 0, 0],
                 [0, cosx, -sinx],
                 [0, sinx, cosx]]))
    if Ms:
        return reduce(np.dot, Ms[::-1])
    return np.eye(3)


def rot_angle_axis(angle, axis):
    """
    Returns a 3x3 rotation matrix that performs a rotation around axis by angle
    """
    norm_axis = np.linalg.norm(axis)
    if norm_axis == 0:
        return np.eye(3)
    u = axis / norm_axis
    cosval, sinval = np.cos(angle), np.sin(angle)
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                                [u[2], 0.0, -u[0]],
                                [-u[1], u[0], 0.0]])
    R = cosval * np.eye(3) + sinval * cross_prod_mat + (1.0 - cosval) * np.outer(u, u)

    return R