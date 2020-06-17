#!/usr/bin/env python

from RobotArm import *
import rospy
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Header
import numpy as np
class PublishStateNode:
    def __init__(self):
        self.arm = RobotArm(0, 10.4, 8.9, 17.5)
        isAvailable = self.arm.move_lerp((15,0,0),(15,0,30))
        self.comm(isAvailable[1])

    def comm(self, frameLists):
        jointPub = rospy.Publisher("joint_states", JointState, queue_size=10)
        markerPub = rospy.Publisher("robotMarker", Marker, queue_size=100)
        rospy.init_node('joint_state_publisher')
        rate = rospy.Rate(10)
        js = JointState()
        js.header = Header()
        mk = Marker()
        idx = 0
        points = []
        while not rospy.is_shutdown() and (idx < len(frameLists)):
            js.header.stamp = rospy.Time.now()
            js.name = ['floor_to_link_0', 
                        'link_0_to_link_1', 
                        'link_1_to_link_2', 
                        'link_2_to_link_3',
                        'link_3_to_gripper_pole', 
                        'left_gripper_joint', 
                        'right_gripper_joint']
            
            position_list = list(reversed(frameLists[idx]))
            position_list.append(position_list[len(position_list)-1])
            position_list[1] -= np.pi / 2
            js.position = position_list
            js.velocity = []
            js.effort = []

            mk.header.frame_id = "floor"
            mk.header.stamp = rospy.Time.now()
            mk.ns = "my_namespace"
            mk.id = 0
            mk.type = Marker.LINE_STRIP
            mk.action = Marker.ADD
            thetas = list(reversed(frameLists[idx][2:]))
            thetas.insert(4, 0)
            endPoint = self.arm.forward_kinematics(thetas)[:3, 3]
            p = Point()
            p.x = endPoint[1]/-100.0
            p.y = endPoint[0]/100.0
            p.z = endPoint[2]/100.0+0.09
            points.append(p)
            mk.points = points
            mk.pose.orientation.w = 1.0
            mk.scale.x = 0.01
            mk.color.a = 1.0 # Don't forget to set the alpha!
            mk.color.g = 1.0
            markerPub.publish(mk)

            jointPub.publish(js)
            idx += 1
            if idx == len(frameLists):
                points = []
                idx = 0
            rate.sleep()

if __name__ == '__main__':
    PublishStateNode()