from pc_util import (normalize_to_unit_sphere, euler2mat,
                    in_range, bresenham_circle_alpha_disk )
import numpy as np

class PC_Render():
    def __init__(self,
                pointcloud,
                rgb             =   [99, 184, 255],
                alpha           =   0.5,
                bg_color        =   [255,255,255],
                rot             =   [0, 0, 0],
                canvas_size     =   500,    # pixels
                paint_size      =   200,
                diameter        =   20,
                depth_decrease  =   0.5,
                light           =   True,
                normalize       =   True   ):

        if pointcloud is None or pointcloud.shape[0] == 0:
            raise ValueError
        if normalize:
            pointcloud = normalize_to_unit_sphere(pointcloud)
        self.pc = pointcloud

        if isinstance(rot, list):
            rot = euler2mat(*rot)
        self.rot = np.array(rot)

        self.rgb = np.array(rgb)
        self.bg_color = np.array(bg_color)
        self.alpha      = alpha
        self.depth_dec  = depth_decrease
        self.diameter   = diameter
        self.light      = light
        self.paint_size = paint_size

        if isinstance(canvas_size, int):
            self.canvas_w = canvas_size
            self.canvas_h = canvas_size
        else:
            self.canvas_w = canvas_size[0]
            self.canvas_h = canvas_size[1]

        # color_disk, alpha_disk, depth_disk, disk_dx, disk_dy
        self.update_disk()

    def update(self, diameter=None, canvas_size=None, paint_size=None, rot=None):
        if diameter is not None:
            self.diameter = diameter
            self.update_disk()
        if canvas_size is not None:
            self.canvas_w = canvas_size[0]
            self.canvas_h = canvas_size[1]
        if paint_size is not None:
            self.paint_size = paint_size
        if rot is not None:
            self.rot = np.array(rot)

    def update_disk(self):
        """ Pre-compute the Gaussian disk
            color_disk, alpha_disk, depth_disk, disk_dx, disk_dy
        """
        alpha_disk = bresenham_circle_alpha_disk(self.diameter) # [D, D], val 0~1
        disk_x, disk_y = (alpha_disk > 0).nonzero()
        self.alpha_disk = alpha_disk[disk_x, disk_y]
        # map disk_xy to center of circle
        radius = (self.diameter-1)/2
        self.disk_x = disk_x - radius
        self.disk_y = disk_y - radius
        if self.light:
            delta = 1/3 * radius
            self.color_disk = np.exp((-(self.disk_x+delta)**2 - (self.disk_y-delta)**2)/(radius**2))
        else:
            self.color_disk = np.exp((-self.disk_x**2 - self.disk_y**2)/(radius**2))
        self.depth_disk = -np.sqrt(np.maximum(radius**2 - self.disk_x**2 - self.disk_y**2, 0))
        self.disk_x = self.disk_x.astype(np.int)
        self.disk_y = self.disk_y.astype(np.int)

    def draw(self):
        # init canvas
        img_canvas = np.full((self.canvas_w, self.canvas_h, 3), self.bg_color)
        img_depth  = np.full((self.canvas_w, self.canvas_h), np.inf)

        # order pc by z, from zmin to zmax: depth_factor = 1 ~ depth_decrease
        pc = (np.dot(self.rot, self.pc.transpose())).transpose()
        if pc.shape[0] > 1 :
            zorder = np.argsort(pc[:, 2])
            pc = pc[zorder, :]
            zmax, zmin = pc[-1, 2], pc[0, 2]
            depth_factor = (zmax - pc[:, 2]) / (zmax - zmin) * (1-self.depth_dec) + self.depth_dec
        else:
            depth_factor = [1,]
        if len(self.rgb.shape) == 2:
            rgb = self.rgb[zorder, :]

        # draw points
        half_w = self.canvas_w//2
        half_h = self.canvas_h//2
        radius = self.diameter//2
        disk_x = self.disk_x + half_w
        disk_y = self.disk_y + half_h
        for i in range(pc.shape[0]-1, -1, -1):
            x_point = int(pc[i, 0] * self.paint_size)
            y_point = int(pc[i, 1] * self.paint_size)
            # ball outside canvas
            if not in_range(x_point, [-half_w - radius, half_w + radius]) or \
               not in_range(y_point, [-half_h - radius, half_h + radius]):
                continue
            x_abs = disk_x + x_point
            y_abs = disk_y + y_point

            # ball on edge
            if not in_range(x_point, [-half_w + radius, half_w - radius]) or \
               not in_range(y_point, [-half_h + radius, half_h - radius]):
                xy_mask = np.logical_and(np.logical_and(x_abs>=0,  x_abs<self.canvas_w),
                                        np.logical_and(y_abs>=0,  y_abs<self.canvas_h))
                x_abs = x_abs[xy_mask]
                y_abs = y_abs[xy_mask]
                depth_disk = self.depth_disk[xy_mask]
                color_disk = self.color_disk[xy_mask]
                alpha_disk = self.alpha_disk[xy_mask]
            else:
                depth_disk = self.depth_disk
                color_disk = self.color_disk
                alpha_disk = self.alpha_disk
            # check depth
            depth = depth_disk+pc[i, 2]*self.paint_size
            depth_mask = (depth < img_depth[x_abs, y_abs])
            x_abs = x_abs[depth_mask]
            y_abs = y_abs[depth_mask]
            color_disk = color_disk[depth_mask]
            alpha_disk = alpha_disk[depth_mask]
            img_depth[x_abs, y_abs] = depth[depth_mask]
            # draw
            color = self.rgb if len(self.rgb.shape)==1 else rgb[i]
            color_disk = color_disk.reshape(-1,1).repeat(3, axis=1)
            alpha_disk = alpha_disk.reshape(-1,1).repeat(3, axis=1)
            front_color = color_disk * depth_factor[i] * color
            img_canvas[x_abs, y_abs] = img_canvas[x_abs, y_abs]*(1-alpha_disk) + front_color * alpha_disk

        return np.uint8(img_canvas)