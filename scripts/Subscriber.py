#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import cv2
import os
from Vision import Vision

class Subscriber:
    def __init__(self):
        self.bridge = CvBridge()
        # self.subscriber = rospy.Subscriber(topicName, Image, self.callback)
        self.image_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
        # self.depth_sub = message_filters.Subscriber("/camera/depth/color/points", PointCloud2, self.callaback)
        self.depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)
        self.ts = message_filters.TimeSynchronizer([self.image_sub, self.depth_sub], 10)
        self.ts.registerCallback(self.callback)
        self.vision = Vision()

    def callback(self, data, data2):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as error:
            rospy.logerr(error)
        
        try:
            cv_depth_image = self.bridge.imgmsg_to_cv2(data2, desired_encoding="16UC1")
        except CvBridgeError as error:
            rospy.logerr(error)

        
        # cv2.imshow("Image Window", cv_image)
        print("Subscribed")
        self.vision.mainLoop(cv_image, cv_depth_image)

        # dirName = "/home/charlie/Project/RobotArm/src/arm_control/scripts/calib_images/"
        # idx = len([name for name in os.listdir(dirName) if name.endswith("jpg")])
        # idx = str(idx) if idx >= 10 else "0" + str(idx)
        # if not os.path.isfile(dirName+idx+".jpg"):
        #     print("saved")
        #     cv2.imwrite(dirName+idx+".jpg", cv_image)
            

        
if __name__ == "__main__":
    sub = Subscriber()
    rospy.init_node('subscriber', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting Down")
    cv2.destroyAllWindows()