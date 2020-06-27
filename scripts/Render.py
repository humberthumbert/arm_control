#!/usr/bin/env python
# -*- coding: utf-8 -*-

from OpenGL.GL import *
# from OpenGL.GL import shaders
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.arrays import vbo
from OpenGLContext import *
import glm
import numpy as np
from ctypes import *

import rospy
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import cv2
import os

from Shader import Shader

def glm_mat_to_numpy(glm_mat):
    numpy_mat = np.zeros((len(glm_mat),len(glm_mat[0])))
    for i in range(len(glm_mat)):
        for j in range(len(glm_mat[i])):
            numpy_mat[i][j] = glm_mat[i][j]
    return numpy_mat 

class Scene:
    def __init__(self):
        self.IS_PERSPECTIVE = True                               # 透视投影
        self.VIEW = np.array([-0.8, 0.8, -0.8, 0.8, 1.0, 20.0])  # 视景体的left/right/bottom/top/near/far六个面
        self.SCALE_K = np.array([1.0, 1.0, 1.0])                 # 模型缩放比例
        self.EYE = np.array([0.0, 0.0, 2.0])                     # 眼睛的位置（默认z轴的正方向）
        self.LOOK_AT = np.array([0.0, 0.0, 0.0])                 # 瞄准方向的参考点（默认在坐标原点）
        self.EYE_UP = np.array([0.0, 1.0, 0.0])                  # 定义对观察者而言的上方（默认y轴的正方向）
        self.WIN_W, self.WIN_H = 640, 480                             # 保存窗口宽度和高度的变量
        self.LEFT_IS_DOWNED = False                              # 鼠标左键被按下
        self.MOUSE_X, self.MOUSE_Y = 0, 0                             # 考察鼠标位移量时保存的起始位置

        self.DIST, self.PHI, self.THETA = self.getposture()       # 眼睛与观察目标之间的距离、仰角、方位角
        self.vertices_list = []                              # VBO buffer
        self.indices_list = []                                   # VBO indices


    def getposture(self):
        phi = 0.0
        theta = 0.0
        dist = np.sqrt(np.power((self.EYE-self.LOOK_AT), 2).sum())
        if dist > 0:
            phi = np.arcsin((self.EYE[1]-self.LOOK_AT[1])/dist)
            theta = np.arcsin((self.EYE[0]-self.LOOK_AT[0])/(dist*np.cos(phi)))
            return dist, phi, theta

        return dist, phi, theta
        
    def init(self):
        glClearColor(0.0, 0.0, 0.0, 1.0) # 设置画布背景色。注意：这里必须是4个参数
        glEnable(GL_DEPTH_TEST)          # 开启深度测试，实现遮挡关系
        glDepthFunc(GL_LEQUAL)           # 设置深度测试函数（GL_LEQUAL只是选项之一）
        folderPath = os.path.dirname(os.path.abspath(__file__))
        self.shader = Shader(os.path.join(folderPath, "vertex.glsl"), 
                        os.path.join(folderPath, "fragment.glsl"))

    def draw(self):
        # 清除屏幕及深度缓存
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # 设置投影（透视投影）
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        if self.WIN_W > self.WIN_H:
            if self.IS_PERSPECTIVE:
                glFrustum(self.VIEW[0]*self.WIN_W/self.WIN_H, self.VIEW[1]*self.WIN_W/self.WIN_H, self.VIEW[2], self.VIEW[3], self.VIEW[4], self.VIEW[5])
            else:
                glOrtho(self.VIEW[0]*self.WIN_W/self.WIN_H, self.VIEW[1]*self.WIN_W/self.WIN_H, self.VIEW[2], self.VIEW[3], self.VIEW[4], self.VIEW[5])
        else:
            if self.IS_PERSPECTIVE:
                glFrustum(self.VIEW[0], self.VIEW[1], self.VIEW[2]*self.WIN_H/self.WIN_W, self.VIEW[3]*self.WIN_H/self.WIN_W, self.VIEW[4], self.VIEW[5])
            else:
                glOrtho(self.VIEW[0], self.VIEW[1], self.VIEW[2]*self.WIN_H/self.WIN_W, self.VIEW[3]*self.WIN_H/self.WIN_W, self.VIEW[4], self.VIEW[5])
            
        # 设置模型视图
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
            
        # 几何变换
        glScale(self.SCALE_K[0], self.SCALE_K[1], self.SCALE_K[2])
            
        # 设置视点
        gluLookAt(
            self.EYE[0], self.EYE[1], self.EYE[2], 
            self.LOOK_AT[0], self.LOOK_AT[1], self.LOOK_AT[2],
            self.EYE_UP[0], self.EYE_UP[1], self.EYE_UP[2]
        )
        
        # 设置视口
        glViewport(0, 0, self.WIN_W, self.WIN_H)

        # add coordinate
        self.draw_coordinate()
        self.draw_vbo()

        # ---------------------------------------------------------------
        glutSwapBuffers()                    # 切换缓冲区，以显示绘制内容
        
    def reshape(self, width, height):

        self.WIN_W, self.WIN_H = width, height
        glutPostRedisplay()
        
    def mouseclick(self, button, state, x, y):

        self.MOUSE_X, self.MOUSE_Y = x, y
        if button == GLUT_LEFT_BUTTON:
            self.LEFT_IS_DOWNED = state==GLUT_DOWN
        elif button == 3:
            self.SCALE_K *= 1.05
            glutPostRedisplay()
        elif button == 4:
            self.SCALE_K *= 0.95
            glutPostRedisplay()
        
    def mousemotion(self, x, y):

        if self.LEFT_IS_DOWNED:
            dx = self.MOUSE_X - x
            dy = y - self.MOUSE_Y
            self.MOUSE_X, self.MOUSE_Y = x, y
            
            self.PHI += 2*np.pi*dy/self.WIN_H
            self.PHI %= 2*np.pi
            self.THETA += 2*np.pi*dx/self.WIN_W
            self.THETA %= 2*np.pi
            r = self.DIST*np.cos(self.PHI)
            
            self.EYE[1] = self.DIST*np.sin(self.PHI)
            self.EYE[0] = r*np.sin(self.THETA)
            self.EYE[2] = r*np.cos(self.THETA)
                
            if 0.5*np.pi < self.PHI < 1.5*np.pi:
                self.EYE_UP[1] = -1.0
            else:
                self.EYE_UP[1] = 1.0
            
            glutPostRedisplay()
        
    def keydown(self, key, x, y):

        if key in [b'x', b'X', b'y', b'Y', b'z', b'Z']:
            if key == b'x': # 瞄准参考点 x 减小
                self.LOOK_AT[0] -= 0.01
            elif key == b'X': # 瞄准参考 x 增大
                self.LOOK_AT[0] += 0.01
            elif key == b'y': # 瞄准参考点 y 减小
                self.LOOK_AT[1] -= 0.01
            elif key == b'Y': # 瞄准参考点 y 增大
                self.LOOK_AT[1] += 0.01
            elif key == b'z': # 瞄准参考点 z 减小
                self.LOOK_AT[2] -= 0.01
            elif key == b'Z': # 瞄准参考点 z 增大
                self.LOOK_AT[2] += 0.01
            
            self.DIST, self.PHI, self.THETA = self.getposture()
            glutPostRedisplay()
        elif key == b'\r': # 回车键，视点前进
            self.EYE = self.LOOK_AT + (self.EYE - self.LOOK_AT) * 0.9
            self.DIST, self.PHI, self.THETA = self.getposture()
            glutPostRedisplay()
        elif key == b'\x08': # 退格键，视点后退
            self.EYE = self.LOOK_AT + (self.EYE - self.LOOK_AT) * 1.1
            self.DIST, self.PHI, self.THETA = self.getposture()
            glutPostRedisplay()
        elif key == b' ': # 空格键，切换投影模式
            self.IS_PERSPECTIVE = not self.IS_PERSPECTIVE 
            glutPostRedisplay()

    def render(self): 
        glutInit()
        displayMode = GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH
        glutInitDisplayMode(displayMode)

        glutInitWindowSize(self.WIN_W, self.WIN_H)
        glutInitWindowPosition(300, 200)
        glutCreateWindow('Quidam Of OpenGL')
        
        self.init()                              # 初始化画布
        glutDisplayFunc(self.draw)               # 注册回调函数draw()
        glutReshapeFunc(self.reshape)            # 注册响应窗口改变的函数reshape()
        glutMouseFunc(self.mouseclick)           # 注册响应鼠标点击的函数mouseclick()
        glutMotionFunc(self.mousemotion)         # 注册响应鼠标拖拽的函数mousemotion()
        glutKeyboardFunc(self.keydown)           # 注册键盘输入的函数keydown()
        
        glutMainLoop()

    def draw_coordinate(self):
        # ---------------------------------------------------------------
        glBegin(GL_LINES)                    # 开始绘制线段（世界坐标系）
        
        # 以红色绘制x轴
        glColor4f(1.0, 0.0, 0.0, 1.0)        # 设置当前颜色为红色不透明
        glVertex3f(-0.8, 0.0, 0.0)           # 设置x轴顶点（x轴负方向）
        glVertex3f(0.8, 0.0, 0.0)            # 设置x轴顶点（x轴正方向）
        
        # 以绿色绘制y轴
        glColor4f(0.0, 1.0, 0.0, 1.0)        # 设置当前颜色为绿色不透明
        glVertex3f(0.0, -0.8, 0.0)           # 设置y轴顶点（y轴负方向）
        glVertex3f(0.0, 0.8, 0.0)            # 设置y轴顶点（y轴正方向）
        
        # 以蓝色绘制z轴
        glColor4f(0.0, 0.0, 1.0, 1.0)        # 设置当前颜色为蓝色不透明
        glVertex3f(0.0, 0.0, -0.8)           # 设置z轴顶点（z轴负方向）
        glVertex3f(0.0, 0.0, 0.8)            # 设置z轴顶点（z轴正方向）
        
        glEnd()                              # 结束绘制线段
    
    def draw_vbo(self):
        # glColor4f(0.0, 1.0, 1.0, .5)

        for idx in range(len(self.vertices_list)):
            vao = GLuint(0)
            glGenVertexArrays(1, vao)
            glBindVertexArray(vao)
            print("Loop: {}".format(idx))
            vertices = self.vertices_list[idx]
            indices = self.indices_list[idx]
            vb = GLuint(idx)
            glGenBuffers(1, vb)
            glBindBuffer(GL_ARRAY_BUFFER, vb)
            glBufferData(GL_ARRAY_BUFFER, sys.getsizeof(vertices), vertices, GL_STATIC_DRAW)

            eb = GLuint(idx)
            glGenBuffers(1, eb)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eb)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, sys.getsizeof(indices), indices, GL_STATIC_DRAW)
            # Bind VBO data to VAO
            glVertexAttribPointer(0,3,GL_FLOAT, False, 3*4, None)
            glEnableVertexAttribArray(0)
            # Shader
            # self.shader.use()
            
            view = glm.lookAt(glm.vec3(self.EYE[0], self.EYE[1], self.EYE[2]),
                                glm.vec3(self.LOOK_AT[0], self.LOOK_AT[1], self.LOOK_AT[2]), 
                                glm.vec3(self.EYE_UP[0], self.EYE_UP[1], self.EYE_UP[2]))
            view_mat = glm_mat_to_numpy(view)
            # self.shader.setMat4("View", view_mat)

            glBindVertexArray(vao)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glDrawElements(
                GL_TRIANGLES,
                len(indices),
                GL_UNSIGNED_INT,
                None
            )
            glBindVertexArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    def add_vbo(self, vertex_array, index_array):
        if len(self.vertices_list)==0:
            self.vertices_list = vertex_array
            self.indices_list = index_array
        else:        
            self.vertices_list = np.vstack((self.vertices_list, vertex_array))
            self.indices_list = np.vstack((self.indices_list, index_array))


if __name__ == "__main__":
    scene = Scene()

    vertices = np.array([
        0.5, 0.5, 0.0,
        0.5, -0.5, 0.0,
        -0.5, -0.5, 0.0,
        -0.5, 0.5, 0.0], dtype=np.float32)
    indices = np.array([
        0, 1, 3,
        1, 2, 3
    ], dtype=np.int32)

    scene.add_vbo(vertices, indices)

    vertices = np.array([
        0.2, 0.2, 0.0,
        0.2, -0.2, 0.0,
        -0.2, -0.2, 0.0,
        -0.2, 0.2, 0.0], dtype='f')
    indices = np.array([
        0, 1, 2,
        2, 0, 3
    ], dtype=np.int32)

    scene.add_vbo(vertices, indices)
    scene.render()
