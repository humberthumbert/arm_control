from matplotlib import pyplot as plt
import numpy as np
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
        """
        Calculate the end effector's position and rotation
        """
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

    def forward_kinematics_links_end(self, thetas):
        """
        Calculate and return each link's end position
        """
        if len(thetas) != len(self.links):
            return None

        pos = np.diag([1,1,1,1])
        linkEndPositions = []
        for i in range(len(thetas)):
            link = self.links[i]
            theta = thetas[i]
            A = link.A(theta)
            pos = np.dot(pos, A)
            linkEndPositions.append(np.squeeze(np.asarray(pos[0:3,3])).tolist())
        return linkEndPositions

    def jacobian(self, thetas):
        """
            Get the jacobian matrix with the position of endEffector
        """
        linkEndPositions = self.forward_kinematics_links_end(thetas)
        
        rotation = np.diag([1,1,1])
        endEffectorPos = np.array(linkEndPositions)[-1]
        idx = 0
        jacobian_matrix = np.empty((3, len(self.links)))
        for linkEndPos in linkEndPositions:
            rotation = np.dot(rotation, self.links[idx].mat[:3,:3]) # 3x1 vector
            screw_axis = rotation[:, 2]
            screw_axis = (screw_axis / np.linalg.norm(screw_axis)).T
            j = np.cross(screw_axis, endEffectorPos - linkEndPos)
            jacobian_matrix[:3, idx] = j
            idx += 1

        # print(jacobian_matrix)
        return jacobian_matrix.T, endEffectorPos


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
        ax.set_xlim3d(-axis_max, axis_max)
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
            print("========= Frame {} ========".format(count))
            endPos = np.diag([1,1,1,1])
            positions = [[0,0,0]]
            for i in range(len(thetas)):
                link = self.links[i]
                theta = thetas[len(thetas) - 1 - i]
                A = link.A(theta)
                endPos = np.dot(endPos, A)
                pos = np.squeeze(np.asarray(endPos[0:3,3])).tolist()
                positions.append(pos)
                print("Joint {} position: {} \t {} \t {}".format(i, pos[0], pos[1], pos[2]))
            positions = np.array(positions).T.tolist()
            frame_joint_pos.append(positions)

        def update(num, data, line, trace):
            line.set_data((data[:, 0, num], data[:, 1, num]))
            line.set_3d_properties(data[:, 2, num])
            print(num)
            trace.set_data(data[-1, 0, :num+1], data[-1, 1, :num+1])
            trace.set_3d_properties(data[-1, 2, :num+1])

        N = len(thetaFrames)
        data = frame_joint_pos
        data = np.array(frame_joint_pos).T
        print(data[:, 0, :3], data[:, 1, :3], data[:, 2, :3])
        line, = ax.plot(data[:, 0, 0], data[:, 1, 0], data[:, 2, 0], color="black", label="arm")
        trace, = ax.plot(data[-1, 0, 0:1], data[-1, 1, 0:1], data[-1, 2, 0:1])

        # Setting the axes properties
        ax.set_xlim3d([-400.0, 400.0])
        ax.set_xlabel('X')

        ax.set_ylim3d([-400.0, 400.0])
        ax.set_ylabel('Y')

        ax.set_zlim3d([-400.0, 400.0])
        ax.set_zlabel('Z')

        ani = animation.FuncAnimation(fig, update, N, fargs=(data, line, trace), interval=int(10000/N), blit=False)
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

    def plot_thetas(self, thetaFrames):
        fig = plt.figure()
        ax = fig.subplots()
        x = np.array(list(range(len(thetaFrames[:,0]))))
        ax.plot(x, thetaFrames[:, 2],label="theta 3")
        ax.plot(x, thetaFrames[:, 3],label="theta 2")
        ax.plot(x, thetaFrames[:, 4],label="theta 1")
        ax.plot(x, thetaFrames[:, 2:5].sum(axis=1),label="sum")
        ax.legend()
        plt.show()
if __name__ == "__main__":
    DHparams = [
        [96.0355,   0,          0],
        [0,         9.6526,     PI/2], 
        [0,         104.0153,   0],
        [0,         88.6725,    0],
        [0,         170.8423,   0]]
    sl = SerialLink(DHparams)
    # value = np.array([0,0,0,0,PI/2])
    value = np.array([0, 2.2267903161604954, -1.5778460032088937, -1.3261320627254016, 0])
    value = np.array([ 0.80212567,  3.14159265, -1.0926474 , -1.57079633,  0.        ])
    # value = list(reversed(value))
    sl.jacobian(value)
    # sl.jacobian([np.pi/4, np.pi/2, np.pi/4, np.pi/6 , 0])
    sl.plot()