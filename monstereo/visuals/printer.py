"""
Class for drawing frontal, bird-eye-view and combined figures
"""
# pylint: disable=attribute-defined-outside-init
import math
import json
from collections import OrderedDict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse, Circle, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..utils import pixel_to_camera, get_task_error, project


class Printer:
    """
    Print results on images: birds eye view and computed distance
    """
    FONTSIZE_BV = 16
    FONTSIZE = 18
    TEXTCOLOR = 'darkorange'
    COLOR_KPS = 'yellow'

    def __init__(self, image, output_path, kk, output_types, epistemic=False, z_max=30, fig_width=10):

        self.im = image
        self.kk = kk
        self.output_types = output_types
        self.epistemic = epistemic
        self.z_max = z_max  # To include ellipses in the image
        self.y_scale = 1
        self.width = self.im.size[0]
        self.height = self.im.size[1]
        self.fig_width = fig_width

        # Define the output dir
        self.output_path = output_path
        self.cmap = cm.get_cmap('jet')
        self.extensions = []

        # Define variables of the class to change for every image
        self.mpl_im0 = self.stds_ale = self.stds_epi = self.xx_gt = self.zz_gt = self.xx_pred = self.zz_pred =\
            self.dds_real = self.uv_centers = self.uv_shoulders =  self.uv_kps = self.boxes = self.boxes_gt = \
            self.uv_camera = self.radius = self.auxs = None

    def _process_results(self, dic_ann):
        # Include the vectors inside the interval given by z_max
        self.stds_ale = dic_ann['stds_ale']
        self.stds_epi = dic_ann['stds_epi']
        self.gt = dic_ann['gt']  # regulate ground-truth matching
        self.xx_gt = [xx[0] for xx in dic_ann['xyz_real']]
        self.xx_pred = [xx[0] for xx in dic_ann['xyz_pred']]

        self.pos_pred = [xx[0:3] for xx in dic_ann['xyz_pred']]
        self.pos_gt = [xx[0:3] for xx in dic_ann['xyz_real']]


        if "kps_3d_pred" in dic_ann.keys():
            self.kps_3d_pred = np.array(dic_ann['kps_3d_pred'])
            self.kps_3d_conf = np.array(dic_ann['kps_3d_conf'])
            #self.kps_3d_gt = dic_ann['kps_3d_gt']



        try:
            self.angles = dic_ann['angles_egocentric']
        except KeyError:
            print("key error for the angle")

        # Do not print instances outside z_max
        self.zz_gt = [xx[2] if xx[2] < self.z_max - self.stds_epi[idx] else 0
                      for idx, xx in enumerate(dic_ann['xyz_real'])]
        self.zz_pred = [xx[2] if xx[2] < self.z_max - self.stds_epi[idx] else 0
                        for idx, xx in enumerate(dic_ann['xyz_pred'])]

        self.dds_real = dic_ann['dds_real']
        self.uv_shoulders = dic_ann['uv_shoulders']
        self.uv_centers = dic_ann['uv_centers']
        self.boxes = dic_ann['boxes']
        self.boxes_gt = dic_ann['boxes_gt']

        try:
            self.car_models = dic_ann['car_model']
            print("CAR_MODEL PRESENT")
        except KeyError:
            self.car_models = None


        self.uv_camera = (int(self.im.size[0] / 2), self.im.size[1])
        self.radius = 11 / 1600 * self.width
        if dic_ann['aux']:
            self.auxs = dic_ann['aux'] if dic_ann['aux'] else None

    def factory_axes(self):
        """Create axes for figures: front bird combined"""
        axes = []
        figures = []

        #  Initialize combined figure, resizing it for aesthetic proportions
        if 'combined_3d' in self.output_types:
            assert 'bird' and 'front' not in self.output_types, \
                "combined figure cannot be print together with front or bird ones"

            self.y_scale = self.width / (self.height * 2)  # Defined proportion
            if self.y_scale < 0.95 or self.y_scale > 1.05:  # allows more variation without resizing
                self.im = self.im.resize((self.width, round(self.height * self.y_scale)))
            self.width = self.im.size[0]
            self.height = self.im.size[1]
            fig_width = self.fig_width + 0.6 * self.fig_width

            fig_height = 2 * self.fig_width * self.height / self.width

            # Distinguish between KITTI images and general images
            fig_ar_1 = 0.8
            width_ratio = 1.9
            self.extensions.append('.combined_3d.png')

            fig, (ax0s, ax1s) = plt.subplots(2, 2, sharey=False, gridspec_kw={'width_ratios': [2.4, 1.7]},
                                           figsize=(fig_width, fig_height))
            ax0, ax1 = ax0s
            ax2, ax3 = ax1s
            ax3.remove()
            ax3 = fig.add_subplot(2,2,4,projection='3d')

            ax1.set_aspect(fig_ar_1)
            fig.set_tight_layout(True)
            fig.subplots_adjust(left=0.02, right=0.98, bottom=0, top=1, hspace=0, wspace=0.02)

            figures.append(fig)

        elif 'combined_kps' in self.output_types:
            assert 'bird' and 'front' not in self.output_types, \
                "combined figure cannot be print together with front or bird ones"

            self.y_scale = self.width / (self.height * 2)  # Defined proportion
            if self.y_scale < 0.95 or self.y_scale > 1.05:  # allows more variation without resizing
                self.im = self.im.resize((self.width, round(self.height * self.y_scale)))
            self.width = self.im.size[0]
            self.height = self.im.size[1]
            fig_width = self.fig_width + 0.6 * self.fig_width

            fig_height = 2 * self.fig_width * self.height / self.width

            # Distinguish between KITTI images and general images
            fig_ar_1 = 0.8
            width_ratio = 1.9
            self.extensions.append('.combined_kps.png')

            fig = plt.figure(figsize=(fig_width, fig_height))
            gs = fig.add_gridspec(2, 2, figure=fig, width_ratios=[2, 2], height_ratios=[1.25,1])

            ax0 = fig.add_subplot(gs[0,:])
            ax1 = fig.add_subplot(gs[1,0])
            ax2 = ax0
            ax3 = fig.add_subplot(gs[1,1],projection='3d')

            ax1.set_aspect(fig_ar_1)
            fig.set_tight_layout(True)
            fig.subplots_adjust(left=0.02, right=0.98, bottom=0, top=1, hspace=0, wspace=0.02)

            figures.append(fig)

        elif 'combined_nkps' in self.output_types:
            assert 'bird' and 'front' not in self.output_types, \
                "combined figure cannot be print together with front or bird ones"

            self.y_scale = self.width / (self.height * 2)  # Defined proportion
            if self.y_scale < 0.95 or self.y_scale > 1.05:  # allows more variation without resizing
                self.im = self.im.resize((self.width, round(self.height * self.y_scale)))
            self.width = self.im.size[0]
            self.height = self.im.size[1]
            fig_width = self.fig_width + 0.6 * self.fig_width

            fig_height = 2 * self.fig_width * self.height / self.width

            # Distinguish between KITTI images and general images
            fig_ar_1 = 0.8
            width_ratio = 1.9
            self.extensions.append('.combined_nkps.png')

            fig = plt.figure(figsize=(fig_width, fig_height))
            gs = fig.add_gridspec(2, 2, figure=fig, width_ratios=[2, 2], height_ratios=[1.25,1])
            ax0 = fig.add_subplot(gs[0,:])
            ax1 = fig.add_subplot(gs[1,0])
            ax2 = ax0
            ax3 = fig.add_subplot(gs[1,1],projection='3d')

            ax1.set_aspect(fig_ar_1)
            fig.set_tight_layout(True)
            fig.subplots_adjust(left=0.02, right=0.98, bottom=0, top=1, hspace=0, wspace=0.02)

            figures.append(fig)

        elif '3d_visu' in self.output_types:
            assert 'bird' and 'front' not in self.output_types, \
                "combined figure cannot be print together with front or bird ones"

            self.y_scale = self.width / (self.height * 2)  # Defined proportion
            if self.y_scale < 0.95 or self.y_scale > 1.05:  # allows more variation without resizing
                self.im = self.im.resize((self.width, round(self.height * self.y_scale)))
            self.width = self.im.size[0]
            self.height = self.im.size[1]
            fig_width = self.fig_width + 0.6 * self.fig_width

            fig_height = 2 * self.fig_width * self.height / self.width

            # Distinguish between KITTI images and general images
            fig_ar_1 = 0.8
            width_ratio = 1.9
            self.extensions.append('.3d_visu.png')

            fig = plt.figure(figsize=(fig_width, fig_height))
            gs = fig.add_gridspec(2, 2, figure=fig, width_ratios=[2, 2], height_ratios=[1.25,1])

            ax0 = fig.add_subplot(gs[0,:])
            ax1 = fig.add_subplot(gs[1,0])
            ax2 = ax0
            ax3 = fig.add_subplot(gs[1,1],projection='3d')

            ax1.set_aspect(fig_ar_1)
            fig.set_tight_layout(True)
            fig.subplots_adjust(left=0.02, right=0.98, bottom=0, top=1, hspace=0, wspace=0.02)

            figures.append(fig)

        elif 'combined' in self.output_types:
            assert 'bird' and 'front' not in self.output_types, \
                "combined figure cannot be print together with front or bird ones"

            self.y_scale = self.width / (self.height * 2)  # Defined proportion
            if self.y_scale < 0.95 or self.y_scale > 1.05:  # allows more variation without resizing
                self.im = self.im.resize((self.width, round(self.height * self.y_scale)))
            self.width = self.im.size[0]
            self.height = self.im.size[1]
            fig_width = self.fig_width + 0.6 * self.fig_width
            fig_height = self.fig_width * self.height / self.width

            # Distinguish between KITTI images and general images
            fig_ar_1 = 0.8
            width_ratio = 1.9
            self.extensions.append('.combined.png')

            fig, (ax0, ax1) = plt.subplots(1, 2, sharey=False, gridspec_kw={'width_ratios': [width_ratio, 1]},
                                           figsize=(fig_width, fig_height))
            ax1.set_aspect(fig_ar_1)
            fig.set_tight_layout(True)
            fig.subplots_adjust(left=0.02, right=0.98, bottom=0, top=1, hspace=0, wspace=0.02)

            figures.append(fig)
            assert 'front' not in self.output_types and 'bird' not in self.output_types, \
                "--combined arguments is not supported with other visualizations"

        # Initialize front figure
        elif 'front' in self.output_types:
            width = self.fig_width
            height = self.fig_width * self.height / self.width
            self.extensions.append(".front.png")
            plt.figure(0)
            fig0, ax0 = plt.subplots(1, 1, figsize=(width, height))
            fig0.set_tight_layout(True)
            figures.append(fig0)

        # Create front figure axis
        if any(xx in self.output_types for xx in ['front', 'combined', 'combined_3d',
                                                'combined_kps', 'combined_nkps', '3d_visu']):
            ax0 = self.set_axes(ax0, axis=0)

            divider = make_axes_locatable(ax0)
            cax = divider.append_axes('right', size='3%', pad=0.05)
            bar_ticks = self.z_max // 5 + 1
            norm = matplotlib.colors.Normalize(vmin=0, vmax=self.z_max)
            scalar_mappable = plt.cm.ScalarMappable(cmap=self.cmap, norm=norm)
            scalar_mappable.set_array([])
            plt.colorbar(scalar_mappable, ticks=np.linspace(0, self.z_max, bar_ticks),
                         boundaries=np.arange(- 0.05, self.z_max + 0.1, .1), cax=cax, label='Z [m]')

            axes.append(ax0)
        if not axes:
            axes.append(None)

        # Initialize bird-eye-view figure
        if 'bird' in self.output_types:
            self.extensions.append(".bird.png")
            fig1, ax1 = plt.subplots(1, 1)
            fig1.set_tight_layout(True)
            figures.append(fig1)
        if any(xx in self.output_types for xx in ['bird', 'combined', 'combined_3d', 'combined_kps',
                                                    'combined_nkps', '3d_visu']):
            ax1 = self.set_axes(ax1, axis=1)  # Adding field of view
            axes.append(ax1)

        if any(xx in self.output_types for xx in ['combined_3d', 'combined_kps', 'combined_nkps', '3d_visu']):
            ax2 =self.set_axes(ax2, axis = 2)
            ax3 =self.set_axes(ax3, axis = 3)
            axes.append(ax2)
            axes.append(ax3)

        return figures, axes

    def draw(self, figures, axes, dic_out, image, show_all=False, draw_text=True, legend=True, draw_box=False,
             save=False, show=False, kps=None):

        draw_box = False
        keypoints = []
        if any(xx in self.output_types for xx in ['combined_3d', 'combined_kps', 'combined_nkps', '3d_visu']):
            _, _, pifpaf_out = kps[:]

            for pifpaf_o in pifpaf_out:
                #keypoints.append(np.reshape(pifpaf_o['keypoints'], (3,-1)))
                length = len(pifpaf_o['keypoints'])/3
                x = pifpaf_o['keypoints'][0::3]
                y = pifpaf_o['keypoints'][1::3]
                keypoints.append([x,y])
        # Process the annotation dictionary of monoloco
        self._process_results(dic_out)

        # whether to include instances that don't match the ground-truth
        iterator = range(len(self.zz_pred)) if show_all else range(len(self.zz_gt))
        if not iterator:
            print("-"*110 + '\n' + "! No instances detected, be sure to include file with ground-truth values or "
                  "use the command --show_all" + '\n' + "-"*110)

        # Draw the front figure
        num = 0
        self.mpl_im0.set_data(image)
        for idx in iterator:
            if any(xx in self.output_types for xx in ['front', 'combined', 'combined_3d', 'combined_kps',
                                                    'combined_nkps', '3d_visu']) and self.zz_pred[idx] > 0:

                color = self.cmap((self.zz_pred[idx] % self.z_max) / self.z_max)
                #color = 'red'
                self.draw_circle(axes, self.uv_shoulders[idx], color)
                if draw_box:
                    self.draw_boxes(axes, idx, color)

                if draw_text:
                    self.draw_text_front(axes, self.uv_shoulders[idx], num)
                    num += 1

        # Draw the bird figure
        num = 0

        for idx in iterator:
            if any(xx in self.output_types for xx in ['bird', 'combined', 'combined_3d',
                                                        'combined_kps', 'combined_nkps', '3d_visu']) and self.zz_pred[idx] > 0:

                # Draw ground truth and uncertainty
                self.draw_uncertainty(axes, idx)

                # Draw bird eye view text
                if draw_text:
                    self.draw_text_bird(axes, idx, num, angle = False)
                    num += 1
        # Add the legend

        previous_idx = []
        for idx in iterator:
            if any(xx in self.output_types for xx in ['combined_3d','combined_kps', 'combined_nkps', '3d_visu']):
                #self.draw_keypoints(axes, keypoints[idx])
                previous_idx.append(idx)
                #self.draw_3d_visu(axes, idx)

        for idx, keypoint in enumerate(keypoints):
            if any(xx in self.output_types for xx in ['combined_3d', 'combined_kps', 'combined_nkps', '3d_visu']):
                if idx not in previous_idx:
                    self.draw_3d_visu(axes, idx, color = 'red')

        #? Quick fix for a previous problem. Will be moidifeid in due time
        for idx in iterator:
            if any(xx in self.output_types for xx in ['combined_3d','combined_kps', 'combined_nkps', '3d_visu']):
                #self.draw_keypoints(axes, keypoints[idx])
                self.draw_3d_visu(axes, idx)


        for idx, keypoint in enumerate(keypoints):
            if any(xx in self.output_types for xx in ['combined_3d', 'combined_kps', '3d_visu']):
                self.draw_keypoints(axes, keypoint)
     

        for idx, keypoint in enumerate(keypoints):
            if any(xx in self.output_types for xx in ['combined_3d', 'combined_kps',
                                                     'combined_nkps', '3d_visu']):
                if idx not in previous_idx:
                    #self.draw_missing_pos(axes, idx)
                    self.draw_missing_ellipses(axes, idx)

        for idx, _ in enumerate(keypoints):
            if idx not in previous_idx:
                if any(xx in self.output_types for xx in [ '3d_visu']):
                    self.draw_3d_scatter(axes, idx, color = 'green')

        for idx in iterator:
            if any(xx in self.output_types for xx in [ '3d_visu']):
                self.draw_3d_scatter(axes, idx, color = 'blue')
        if legend:
            draw_legend(axes)

        # Draw, save or/and show the figures
        for idx, fig in enumerate(figures):
            fig.canvas.draw()
            if save:
                fig.savefig(self.output_path + self.extensions[idx], bbox_inches='tight')
            if show:
                fig.show()
            plt.close(fig)

    def draw_keypoints(self, axes, keypoints):
        axes[2].scatter(keypoints[0], np.array(keypoints[1])*self.y_scale)

    def draw_3d_visu(self, axes, idx, color = 'grey'):
        if len(self.car_models) !=0:
            car_model = self.car_models[idx]
        else:
            car_model = 'docs/car_model.json'
        with open(car_model) as json_file:
            data = json.load(json_file)
        vertices = np.array(data['vertices'])
        triangles = np.array(data['faces']) - 1

        x,y,z = self.pos_pred[idx]

        if "Camera" in self.output_path:
            T = np.float32([0.0, self.angles[idx] ,0.0 ,x,y,z])
        #if 'kitti' in self.output_path or  "export" in self.output_path:
        else:
            T = np.float32([0.0, self.angles[idx] + np.pi/2 , 0.0 ,x,y,z])
        scale = np.float32([1, 1, 1])
        if z< self.z_max:
            vertices_r = project(T, scale, vertices )

            axes[3].plot_trisurf(vertices_r[:,0], vertices_r[:,2], triangles,
                                 -vertices_r[:,1] + np.mean(vertices_r[:,1]), shade=True, color=color)

        axes[3].set_xlim([-20, 20])
        axes[3].set_ylim([0, self.z_max])
        axes[3].set_zlim([0, 10])
        axes[3].set_xlabel("X")
        axes[3].set_ylabel("Y")
        axes[3].set_zlabel("Z")

    def draw_3d_scatter(self, axes, idx, color = 'grey'):

        if (self.zz_pred[idx]< self.z_max and self.zz_pred[idx]>0) :

            mask = self.kps_3d_conf[idx]>0
            axes[3].scatter(self.kps_3d_pred[idx][mask,0], self.kps_3d_pred[idx][mask,2],
                             self.kps_3d_pred[idx][mask,1] - np.mean(self.kps_3d_pred[idx][mask,1]  )
                             ,color=color)

        axes[3].set_xlim([-20, 20])
        axes[3].set_ylim([0, self.z_max])
        axes[3].set_zlim([0, 10])
        axes[3].set_xlabel("X")
        axes[3].set_ylabel("Y")
        axes[3].set_zlabel("Z")

    def draw_uncertainty(self, axes, idx):

        theta = math.atan2(self.zz_pred[idx], self.xx_pred[idx])
        dic_std = {'ale': self.stds_ale[idx], 'epi': self.stds_epi[idx]}
        dic_x, dic_y = {}, {}

        # Aleatoric and epistemic
        for key, std in dic_std.items():
            delta_x = std * math.cos(theta)
            delta_z = std * math.sin(theta)
            dic_x[key] = (self.xx_pred[idx] - delta_x, self.xx_pred[idx] + delta_x)
            dic_y[key] = (self.zz_pred[idx] - delta_z, self.zz_pred[idx] + delta_z)

        # MonoLoco
        if not self.auxs:
            axes[1].plot(dic_x['epi'], dic_y['epi'], color='coral', linewidth=2, label="Epistemic Uncertainty")
            axes[1].plot(dic_x['ale'], dic_y['ale'], color='deepskyblue', linewidth=4, label="Aleatoric Uncertainty")
            axes[1].plot(self.xx_pred[idx], self.zz_pred[idx], color='cornflowerblue', label="Prediction", markersize=6,
                         marker='o')
            if self.gt[idx]:
                axes[1].plot(self.xx_gt[idx], self.zz_gt[idx],
                             color='k', label="Ground-truth", markersize=8, marker='x')

        # MonStereo(stereo case)
        elif self.auxs[idx] > 0.5:
            axes[1].plot(dic_x['ale'], dic_y['ale'], color='r', linewidth=4, label="Prediction (mono)")
            axes[1].plot(dic_x['ale'], dic_y['ale'], color='deepskyblue', linewidth=4, label="Prediction (stereo+mono)")
            if self.gt[idx]:
                axes[1].plot(self.xx_gt[idx], self.zz_gt[idx],
                             color='k', label="Ground-truth", markersize=8, marker='x')

        # MonStereo (monocular case)
        else:
            axes[1].plot(dic_x['ale'], dic_y['ale'], color='deepskyblue', linewidth=4, label="Prediction (stereo+mono)")
            axes[1].plot(dic_x['ale'], dic_y['ale'], color='r', linewidth=4, label="Prediction (mono)")
            if self.gt[idx]:
                axes[1].plot(self.xx_gt[idx], self.zz_gt[idx],
                             color='k', label="Ground-truth", markersize=8, marker='x')

    def draw_missing_pos(self,axes, idx):
        if self.zz_pred[idx]<=self.z_max and self.zz_pred[idx] > 0:
            axes[1].plot(self.xx_pred[idx], self.zz_pred[idx], color='red',
                        label="Prediction without GT", markersize=5, marker='o')


    def draw_missing_ellipses(self, axes, idx):
        """draw uncertainty ellipses"""
        if self.zz_pred[idx]<=self.z_max and self.zz_pred[idx] > 0:
            #print(self.zz_pred[idx])
            angle = get_angle(self.xx_pred[idx], self.zz_pred[idx])
            ellipse_ale = Ellipse((self.xx_pred[idx], self.zz_pred[idx]), width=self.stds_ale[idx] * 2,
                                height=1, angle=angle, color='orange',
                                fill=False, label="Aleatoric Uncertainty without GT", linewidth=1.3)
            ellipse_var = Ellipse((self.xx_pred[idx], self.zz_pred[idx]), width=self.stds_epi[idx] * 2,
                                height=1, angle=angle, color='r', fill=False, label="Uncertainty without GT",
                                linewidth=1, linestyle='--')

            axes[1].add_patch(ellipse_ale)
            if self.epistemic:
                axes[1].add_patch(ellipse_var)

            axes[1].plot(self.xx_pred[idx], self.zz_pred[idx], color='red', label="Prediction without GT",
                         markersize=5, marker='o')


    def draw_ellipses(self, axes, idx):
        """draw uncertainty ellipses"""
        target = get_task_error(self.dds_real[idx])
        angle_gt = get_angle(self.xx_gt[idx], self.zz_gt[idx])
        ellipse_real = Ellipse((self.xx_gt[idx], self.zz_gt[idx]), width=target * 2, height=1,
                               angle=angle_gt, color='lightgreen', fill=True, label="Task error")
        axes[1].add_patch(ellipse_real)
        if abs(self.zz_gt[idx] - self.zz_pred[idx]) > 0.001:
            axes[1].plot(self.xx_gt[idx], self.zz_gt[idx], 'kx', label="Ground truth", markersize=3)

        angle = get_angle(self.xx_pred[idx], self.zz_pred[idx])
        ellipse_ale = Ellipse((self.xx_pred[idx], self.zz_pred[idx]), width=self.stds_ale[idx] * 2,
                              height=1, angle=angle, color='b', fill=False, label="Aleatoric Uncertainty",
                              linewidth=1.3)
        ellipse_var = Ellipse((self.xx_pred[idx], self.zz_pred[idx]), width=self.stds_epi[idx] * 2,
                              height=1, angle=angle, color='r', fill=False, label="Uncertainty",
                              linewidth=1, linestyle='--')

        axes[1].add_patch(ellipse_ale)
        if self.epistemic:
            axes[1].add_patch(ellipse_var)

        axes[1].plot(self.xx_pred[idx], self.zz_pred[idx], 'ro', label="Predicted", markersize=3)

    def draw_boxes(self, axes, idx, color):
        ww_box = self.boxes[idx][2] - self.boxes[idx][0]
        hh_box = (self.boxes[idx][3] - self.boxes[idx][1]) * self.y_scale
        ww_box_gt = self.boxes_gt[idx][2] - self.boxes_gt[idx][0]
        hh_box_gt = (self.boxes_gt[idx][3] - self.boxes_gt[idx][1]) * self.y_scale

        rectangle = Rectangle((self.boxes[idx][0], self.boxes[idx][1] * self.y_scale),
                              width=ww_box, height=hh_box, fill=False, color=color, linewidth=3)
        rectangle_gt = Rectangle((self.boxes_gt[idx][0], self.boxes_gt[idx][1] * self.y_scale),
                                 width=ww_box_gt, height=hh_box_gt, fill=False, color='g', linewidth=2)
        axes[0].add_patch(rectangle_gt)
        axes[0].add_patch(rectangle)

    def draw_text_front(self, axes, uv, num):
        axes[0].text(uv[0] + self.radius, uv[1] * self.y_scale - self.radius, str(num),
                     fontsize=self.FONTSIZE, color=self.TEXTCOLOR, weight='bold')

    def draw_text_bird(self, axes, idx, num, angle = False):
        """Plot the number in the bird eye view map"""

        std = self.stds_epi[idx] if self.stds_epi[idx] > 0 else self.stds_ale[idx]
        theta = math.atan2(self.zz_pred[idx], self.xx_pred[idx])


        delta_x = std * math.cos(theta)
        delta_z = std * math.sin(theta)

        if self.zz_pred[idx]+delta_z < self.z_max:

            axes[1].text(self.xx_pred[idx] + delta_x, self.zz_pred[idx] + delta_z,
                        str(num), fontsize=self.FONTSIZE_BV, color='darkorange')
            if angle:
                axes[1].text(self.xx_pred[idx] + delta_x, self.zz_pred[idx] + delta_z-5,
                            str(self.angles[idx]*180/np.pi).split(".")[0],
                            fontsize=self.FONTSIZE_BV, color='black')
        else:
            axes[1].text(self.xx_pred[idx] + delta_x, self.zz_pred[idx] - 5,
                        str(num), fontsize=self.FONTSIZE_BV, color='darkorange')
            if angle:

                axes[1].text(self.xx_pred[idx] + delta_x, self.zz_pred[idx] -5,
                            str(self.angles[idx]*180/np.pi).split(".")[0],
                            fontsize=self.FONTSIZE_BV, color='black')

    def draw_circle(self, axes, uv, color):

        circle = Circle((uv[0], uv[1] * self.y_scale), radius=self.radius, color=color, fill=True)
        axes[0].add_patch(circle)

    def set_axes(self, ax, axis):
        assert axis in (0, 1, 2, 3)

        if axis == 0:
            ax.set_axis_off()
            ax.set_xlim(0, self.width)
            ax.set_ylim(self.height, 0)
            self.mpl_im0 = ax.imshow(self.im)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        if axis == 1:
            uv_max = [0., float(self.height)]
            xyz_max = pixel_to_camera(uv_max, self.kk, self.z_max)
            x_max = abs(xyz_max[0])  # shortcut to avoid oval circles in case of different kk
            corr = round(float(x_max / 3))
            ax.plot([0, x_max], [0, self.z_max], 'k--')
            ax.plot([0, -x_max], [0, self.z_max], 'k--')
            ax.set_xlim(-x_max+corr, x_max-corr)
            ax.set_ylim(0, self.z_max+1)
            ax.set_xlabel("X [m]")

        if axis == 2:
            ax.set_axis_off()
            ax.set_xlim(0, self.width)
            ax.set_ylim(self.height, 0)
            self.mpl_im0 = ax.imshow(self.im)
            #ax.set_ylabel("test_ax_2")


        if axis == 3:
            ax.set_ylabel("test_ax_3")
            print(type(ax))

        return ax


def draw_legend(axes):
    handles, labels = axes[1].get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    axes[1].legend(by_label.values(), by_label.keys(), loc='best')


def get_angle(xx, zz):
    """Obtain the points to plot the confidence of each annotation"""

    theta = math.atan2(zz, xx)
    angle = theta * (180 / math.pi)

    return angle
