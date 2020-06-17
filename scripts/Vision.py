#!/usr/bin/env python

import numpy as np
import cv2
import Calibration
import yaml
import os
import rospy

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
    def mainLoop(self, rgb_image, depth_image):
        dst = cv2.undistort(rgb_image, self.mtx, self.dist, None, self.newcameramatrix)
        x,y,w,h = self.roi
        dst = dst[y:y+h, x:x+w]

        d_image = np.dstack((depth_image, depth_image, depth_image))
        print(depth_image)
        concat_image = cv2.hconcat([dst, rgb_image])
        cv2.imshow("Image Window", depth_image*0.001)
        cv2.waitKey(3)