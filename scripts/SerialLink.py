import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from Link import *

PI = np.pi  

def data_for_sphere(center_x, center_y, center_z, radius):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = radius*np.cos(u)*np.sin(v) + center_x
    y = radius*np.sin(u)*np.sin(v) + center_y
    z = radius*np.cos(v) + center_z
    return x,y,z

class SerialLink(object):
    def __init__(self, DHparams):
        self.links = []
        for DH in DHparams:
            self.links.append(Link(DH[0], DH[1], DH[2]))

    def add_link(self, l, pos=-1):
        assert(type(l) is Link)
        if pos == -1:
            self.links.append(l)
        elif pos >= 0 and pos < len(self.links):
            self.links.insert(pos, l)
        else:
            print("Failed to add link")

    def forward_kinematics(self, thetas):
        if len(thetas) != len(self.links):
            return None

        A_matrice = []
        for i in range(len(thetas)):
            link = self.links[i]
            theta = thetas[i]
            A = link.A(theta)
            A_matrice.append(A)
        
        pos = np.diag([1,1,1,1])
        for A in A_matrice:
            pos = np.dot(pos, A)
        return pos

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        endPos = np.diag([1,1,1,1])
        positions = [[0,0,0]]

        count = 0
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        for link in self.links:
            endPos = np.dot(endPos, link.mat)
            pos = np.squeeze(np.asarray(endPos[0:3,3])).tolist()
            positions.append(pos)

            count += 1
            if (count == len(self.links)):
                break
            x,y,z = data_for_sphere(pos[0], pos[1], pos[2], 0.05)
            ax.plot_wireframe(x, y, z, color=colors[count])
            
        positions = np.array(positions).T
        ax.plot(positions[0], positions[1], positions[2], color="black", label="arm")
        axis_max = np.amax(positions)
        axis_min = np.amin(positions)
        ax.set_xlim3d(axis_min, axis_max)
        ax.set_ylim3d(axis_min, axis_max)
        ax.set_zlim3d(axis_min, axis_max)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.legend()
        plt.show()

    def animate(self, thetaFrames):
        fig = plt.figure()
        ax = Axes3D(fig)

        frame_joint_pos = []
        count = 0
        for thetas in thetaFrames:
            count+=1
            # print("========= Frame {} ========".format(count))
            endPos = np.diag([1,1,1,1])
            positions = []
            for i in range(len(thetas)):
                link = self.links[i]
                theta = thetas[len(thetas) - 1 - i]
                A = link.A(theta)
                endPos = np.dot(endPos, A)
                pos = np.squeeze(np.asarray(endPos[0:3,3])).tolist()
                positions.append(pos)
                # print("Joint {} position: {} \t {} \t {}".format(i, pos[0], pos[1], pos[2]))
            positions = np.array(positions).T.tolist()
            frame_joint_pos.append(positions)

        def update(num, data, line, trace):
            line.set_data((data[:, 0, num], data[:, 1, num]))
            line.set_3d_properties(data[:, 2, num])

            trace.set_data(data[-1, 0, :num+1], data[-1, 1, :num+1])
            trace.set_3d_properties(data[-1, 2, :num+1])

        N = len(thetaFrames)
        data = frame_joint_pos
        data = np.array(frame_joint_pos).T
        print(data[-1, 0, 0:1].tolist(), data[-1, 1, 0:1].tolist(), data[-1, 2, 0:1].tolist())
        print(data[:, 0, 0].tolist(), data[:, 1, 0].tolist(), data[:, 2, 0].tolist())
        line, = ax.plot(data[:, 0, 0], data[:, 1, 0], data[:, 2, 0])
        trace, = ax.plot(data[-1, 0, 0:1], data[-1, 1, 0:1], data[-1, 2, 0:1])

        # Setting the axes properties
        ax.set_xlim3d([-40.0, 40.0])
        ax.set_xlabel('X')

        ax.set_ylim3d([-40.0, 40.0])
        ax.set_ylabel('Y')

        ax.set_zlim3d([-40.0, 40.0])
        ax.set_zlabel('Z')

        animation.FuncAnimation(fig, update, N, fargs=(data, line, trace), interval=int(10000/N), blit=False)
        # ani.save('matplot003.gif', writer='imagemagick')
        plt.show()

    def animate_real(self, opStrings):
        thetaFrames = []
        lastFrameIndex = 0
        for opString in opStrings.split("|")[:-1]:
            signals = list(map(int, opString.split(":")[1].split(";")[:-1]))
            time = int(opString.split(":")[1].split(";")[-1])
            thetas = []
            for i in range(len(signals)):
                if i == 4:
                    theta = -1 * ((signals[i] - 500) / 2000 * PI - PI)
                    thetas.append(theta)
                else:
                    theta = (signals[i] - 500) / 2000 * PI - PI / 2
                    thetas.append(theta)
            if len(thetaFrames) != 0:
                prev_thetas = np.array(thetaFrames[lastFrameIndex])
                diff_thetas = (thetas - prev_thetas) / int(time / 20)
                steps = np.array(list(range(int(time / 20))))[1:]
                steps = diff_thetas.reshape(1,-1) * steps.reshape(-1,1)
                steps = steps + prev_thetas
                for step in steps:
                    thetaFrames.append(step.tolist())
            
            thetaFrames.append(thetas)
            lastFrameIndex = len(thetaFrames) - 1
                
        self.animate(np.array(thetaFrames)[:, 1:])
