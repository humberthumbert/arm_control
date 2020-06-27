#!/usr/bin/env python

import numpy as np
import cv2
import Calibration
import yaml
import os
import rospy

from RobotArm import RobotArm


class Vision:
    def __init__(self):
        dirname = os.path.dirname(os.path.abspath(__file__))
        calib_filename = os.path.join(dirname, "ost.yaml")
        with open(calib_filename, 'r') as f:
            try:
                calib_info = yaml.safe_load(f)
            except yaml.YAMLError as e:
                rospy.logerr(e)
        self.mtx = np.array(calib_info["camera_matrix"]["data"]).reshape(calib_info["camera_matrix"]["rows"], 
                                                                            calib_info["camera_matrix"]["cols"])
        self.ret = np.array(calib_info["projection_matrix"]["data"]).reshape(calib_info["projection_matrix"]["rows"], 
                                                                            calib_info["projection_matrix"]["cols"])
        self.dist = np.array(calib_info["distortion_coefficients"]["data"])
        self.w = calib_info['image_width']
        self.h = calib_info['image_height']
        self.newcameramatrix, self.roi = \
            cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (self.w, self.h), 0, (self.w, self.h))
        self.arm = RobotArm(0, 104.0153, 88.6725, 170.8423, serial_port='/dev/ttyUSB0')
        self.previous_click_pos = []
    def mainLoop(self, rgb_image, depth_image):
        dst = cv2.undistort(rgb_image, self.mtx, self.dist, None, self.newcameramatrix)
        x,y,w,h = self.roi
        dst = dst[y:y+h, x:x+w]

        d_image = np.dstack((depth_image, depth_image, depth_image))
        # print(depth_image)
        # concat_image = cv2.hconcat([dst, rgb_image])
        # rgb_image = detection(rgb_image)

        # Show Image to get point to touch
        cv2.imshow("Image Window", rgb_image)
        cv2.setMouseCallback("Image Window", self.mouse_click)
        cv2.waitKey(3)

    def mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.imageFrameToWorldFrame([0, 112.5, 0], [x, y, 1])

        if event == cv2.EVENT_RBUTTONDBLCLK:
            self.arm.send_command("0:1500;1500;1500;1500;1500;1500;1000|:")


    def imageFrameToWorldFrame(self, camera_pos, click_pos):
        # Intrinsic 3x3
        M1 = self.mtx.copy() 
        invM = np.linalg.inv(M1)
        world_pos = invM * np.matrix(click_pos).T
        print("Click Pixel is {},{}".format(click_pos[0], click_pos[1]))
        x0, y0, z0 = 0,0,0
        x1, y1, z1 = world_pos.T.item(0,0), world_pos.T.item(0,1), world_pos.T.item(0,2)
        x_c = x0 + (-camera_pos[1] - y0) * (x1 - x0) / (y1-y0)
        y_c = -camera_pos[1]
        z_c = z0 + (-camera_pos[1] - y0) * (z1 - z0) / (y1-y0)
        x = 500 + z_c
        y = -x_c
        z = 0
        print("Intersection in world frame {}, {}, {}".format(x, y, z))
        x_robot = x
        y_robot = -y
        z_robot = -100.5
        print("--------------------------------------------------------------------")
        if len(self.previous_click_pos) == 0:
            isAvailable = self.arm.move_to(x_robot, y_robot, z_robot)
            self.arm.send_command(isAvailable[0])
        else:
            isAvailable = self.arm.move_lerp(self.previous_click_pos,
                                                [x_robot, y_robot, z_robot])
            print(isAvailable[0])
            self.arm.send_command(isAvailable[0])
        self.previous_click_pos = (x_robot, y_robot, z_robot)
# rot_x = 0
# rot_y = np.pi/2
# rot_z = np.pi/2
# # Extrinsic 3x4 => 3x3 to planar
# R_z = np.matrix([[np.cos(rot_z), -np.sin(rot_z), 0], 
#                 [np.sin(rot_z), np.cos(rot_z),  0],
#                 [0,             0,              1]])

# R_y = np.matrix([[np.cos(rot_y),    0,  np.sin(rot_y)],
#                 [0,                 1,  0],
#                 [-np.sin(rot_y),    0,  np.cos(rot_y)]])

# R_x = np.matrix([[1,    0,              0],
#                 [0,     np.cos(rot_x),  -np.sin(rot_x)],
#                 [0,     np.sin(rot_x),  np.cos(rot_x)]])
# R = R_z * R_y * R_x
# M2 = np.diag([1.0,1.0,1.0,1.0])
# M2[:3, :3] = R
# M2[:3, 3] = [500, -9.6526, 105-112.5]