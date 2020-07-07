import numpy as np
import matplotlib.pyplot as plt

class Link(object):
    def __init__(self, d, a, alpha, t=0):
        self.d = d  # link offset
        self.a = a  # link length
        self.alpha = alpha  # link twist
        self.t = t  # Joint type, 0 for revolute, 1 for prismatic
        self.theta = 0
        self.mat = self.A(0)

    def A(self, theta):
        self.theta = theta
        self.mat = np.matrix([[np.cos(self.theta),                      -np.sin(self.theta),                    0,                      self.a],
                              [np.sin(self.theta)*np.cos(self.alpha),   np.cos(self.theta)*np.cos(self.alpha),  -np.sin(self.alpha),    -np.sin(self.alpha)*self.d],
                              [np.sin(self.theta)*np.sin(self.alpha),   np.cos(self.theta)*np.sin(self.alpha),  np.cos(self.alpha),     np.cos(self.alpha)*self.d],
                              [0,                                       0,                                      0,                      1]])
        
        return self.mat
    