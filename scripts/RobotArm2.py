import numpy as np
import matplotlib.pyplot as plt
from SerialLink import *
import pyfirmata
import time
import serial
from Interpolation import *

class RobotArm(SerialLink):
    def __init__(self, l0, l1, l2, l3, serial_port=None):
        DHparams = [
            [0, 0, 0],
            [0, l0, PI/2], 
            [0, l1, 0],
            [0, l2, 0],
            [0, l3, 0]]
        super(RobotArm, self).__init__(DHparams)
        self.l0, self.l1, self.l2, self.l3 = (l0, l1, l2, l3)
        if serial_port is not None:
            self.serial = serial.Serial(serial_port, 9600)
        else:
            self.serial = None
        self.lastPose = [1500,1500,1500,1500,1500,1500]

    def reset(self):
        self.serial.write(str.encode("0:1500;1500;1500;1500;1500;1500;1000|:"))

    def inverse_kinematics(self, x, y, z, prev_theta=None):
        thetas = None
        if prev_theta is None:
            for time in range(2):
                for alpha in np.linspace(-PI, PI, num=180*np.power(10,time)+1):
                    thetas = self.inverse_kinematics_2(x, y, z, alpha)
                    if thetas is not None:
                        break
                if thetas is not None:
                    break
        else:
            for time in range(2):
                for alpha_diff in np.linspace(0, 2*PI, num=180*np.power(10,time)+1):
                    alpha = prev_theta + alpha_diff
                    # print(alpha)
                    if alpha <= PI:
                        thetas = self.inverse_kinematics_2(x, y, z, alpha)
                        if thetas is not None:
                            break
                    alpha = prev_theta - alpha_diff
                    if alpha >= -PI:
                        thetas = self.inverse_kinematics_2(x, y, z, alpha)
                        if thetas is not None: 
                            break
                if thetas is not None:
                    break
        return thetas

    def inverse_kinematics_2(self, x, y, z, alpha):
        xy = np.sqrt(x**2 + y**2)
        x_2 = xy + self.l3 * np.cos(alpha+PI)
        y_2 = z + self.l3 * np.sin(alpha+PI)
        c_2 = (x_2**2 + y_2**2 - self.l1**2 - self.l2**2) / (2 * self.l1 * self.l2)
        if np.abs(c_2) > 1:
            # print("Out of range")
            return None
        s_2 = -1 * np.sqrt(1 - c_2**2)
        theta2 = np.arctan2(s_2, c_2)
        k1 = self.l1 + self.l2 * np.cos(theta2)
        k2 = self.l2 * np.sin(theta2)
        theta1 = np.arctan2(y_2, x_2) - np.arctan2(k2, k1)
        theta3 = alpha - theta1 - theta2
        if not self.checkSatisfied(theta1, theta2, theta3, xy, z, alpha):
            s_2 = np.sqrt(1 - c_2**2)
            theta2 = np.arctan2(s_2, c_2)
            k1 = self.l1 + self.l2 * np.cos(theta2)
            k2 = self.l2 * np.sin(theta2)
            theta1 = np.arctan2(y_2, x_2) - np.arctan2(k2, k1)
            theta3 = alpha - theta1 - theta2
            if not self.checkSatisfied(theta1, theta2, theta3, xy, z, alpha):
                # print("Not Satisfied Soltion for ({},{},{}), {}".format(x,y,z,round(alpha/PI*180,2)))
                return None
        if y == 0 and x >= 0:
            theta0 = PI/2
        elif y == 0 and x < 0:
            theta0 = -PI/2
        else:
            theta0 = np.arctan(x/y)

        # self.forward_kinematics([theta0, theta1, theta2, theta3, 0])
        return (theta0, theta1, theta2, theta3, 0, 0)

    def deg_to_signal(self, thetas):
        result = ""
        addon = [PI/2, PI/2, PI/2, PI/2, PI, PI/2]
        min_theta = 2500
        for i in range(len(thetas)):
            theta = thetas[i]
            if i == 4:
                signal = int((2000 * (addon[i]-theta) / PI + 500))
            else:
                signal = int((2000 * (addon[i]+theta) / PI + 500))
            diff_theta = np.abs(signal - self.lastPose[i])
            min_theta = diff_theta if diff_theta < min_theta else min_theta
            result += str(signal) + ";"
        if min_theta < 20:
            min_theta = 20
        result += "200|"#str(int(min_theta)) + "|"
        self.lastPose = thetas
        return result

    def move_to(self, x, y, z, prev_theta=None, move=False):
        thetas = self.inverse_kinematics(x,y,z,prev_theta)
        if thetas is not None:
            thetas = list(reversed(thetas))
            result = "0:" + self.deg_to_signal(thetas) + ":"
            print(result)
            if move and (self.serial is not None):
                self.serial.write(str.encode(result))
            return thetas
        else:
            print("Cannot move to target position")
            return None
            
    def move_with(self, thetas, move=False):
        pos = self.forward_kinematics(thetas)
        if pos is not None:
            result = self.deg_to_signal(thetas)
            print(result)
            if move and (self.serial is not None):
                self.serial.write(str.encode(result))
        else:
            print("Cannot move with required degrees")

    def move_with_signal(self, signals, move=False):
        thetas = (np.array(list(reversed(signals))) * 10 - 500) / 2000 * PI - PI/2
        self.move_with(thetas)
        self.plot()

    def move_along(self, startPos, endPos, move=False):
        midPositions = np.linspace(startPos, endPos, num=21)
        thetas = ""
        index = 0
        midPoses = []
        prev_theta = None
        for midPos in midPositions:
            result = self.move_to(midPos[0], midPos[1], midPos[2], prev_theta)
            # print(result[2])
            if result is None:
                print("Cannot move along this line")
                return None
            else:
                print(result[2])
                # prev_theta = result[2]
                midPoses.append(result)
                result = self.deg_to_signal(result)
                thetas += str(index) + ":" + result
                index += 1
        if move and (self.serial is not None):
            self.serial.write(str.encode(thetas + ":"))
        return (thetas + ":", np.array(midPoses))

    # def move_to_smooth_linear(self, startPos, endPos, time, move=False):
        

    def move_to_smooth_pose(self, startPos, endPos, time, move=False):
        startThetas = self.move_to(startPos[0], startPos[1], startPos[2], move=False)
        endThetas = self.move_to(endPos[0], endPos[1], endPos[2], move=False)
        if (startThetas is None):
            print("Cannot calculate the start pose")
            return
        elif(endThetas is None):
            print("Cannot calculate the end pose")
            return
        # theta(t) = a0 + a1 * t + a2 * t^2 + a3 * t^3
        params = np.zeros((len(startThetas),4))
        for j in range(len(startThetas)):
            params[j][0] = startThetas[j]
            params[j][1] = 0
            params[j][2] = 3 * (endThetas[j] - startThetas[j]) / (time**2)
            params[j][3] = -2 * (endThetas[j] - startThetas[j]) / (time**3)
        
        t = np.linspace(0, time, num=11)
        t = np.array([np.ones(11), np.power(t,1), np.power(t,2), np.power(t,3)])
        midPoses = np.dot(params, t).T # 11xn
        thetas = ""
        index = 0
        for midPose in midPoses:
            result = self.deg_to_signal(midPose)
            # t = int(time/10) if time/10 > 20 else 20
            # result = result.rpartition(";")[0] + ";" + str(int(time/10)) + "|"
            thetas += str(index) + ":" + result
            index += 1

        # Move
        if move and (self.serial is not None):
            self.serial.write(str.encode(thetas + ":"))
        return (thetas + ":", midPoses)

    def move_along_smooth(self, startPos, endPos, time, num, move=False):
        midPositions = np.linspace(startPos, endPos, num=num)
        thetas_commands = ""
        midPoses_list = []
        for i in range(len(midPositions)-1):
            result = self.move_to_smooth_pose(midPositions[i], midPositions[i+1], time=time/(num-1), move=False)
            if result is not None:
                thetas, midPoses = result
                thetas = thetas[:-1]
                thetas_commands += thetas
                midPoses_list.append(midPoses.tolist())

        if move and (self.serial is not None):
            self.serial.write(str.encode(thetas_commands + ":"))
        midPoses = np.array(midPoses_list)
        midPoses = midPoses.reshape(-1, 6)
        return (thetas_commands + ":", midPoses)


    def move_along_smooth_2(self, startPos, endPos, time, num, move=False):
        midPositions = np.linspace(startPos, endPos, num=num)
        times = np.linspace(0, time, num=num)
        return self.move_via_smooth(midPositions, times, move=move)

    def move_via_smooth(self, positions, times, move=False):
        thetas = []
        for i in range(len(positions)):
            result = self.move_to(positions[i][0], positions[i][1], positions[i][2], move=False)
            thetas.append(result)
        results = move_via_smooth_fixed_time(thetas, positions, times)
        results = np.array(results).T
        thetas = ""
        index = 0
        midPoses = []
        for result in results:
            midPoses.append(result)
            result = self.deg_to_signal(result)
            thetas += str(index) + ":" + result

        # Move
        if move and (self.serial is not None):
            self.serial.write(str.encode(thetas + ":"))
        return (thetas + ":", np.array(midPoses))

    def checkIfComplete(self):
        line = self.serial.readline(30)
        if line == b'Poses Complete\r\n':
            return True
        else:
            return False

    def checkSatisfied(self, theta1, theta2, theta3, xy, z, alpha):
        return self.checkIfCorrect(theta1, theta2, theta3, xy, z, alpha) and \
        self.checkIfWithinRange(theta1, theta2, theta3)

    def checkIfCorrect(self, theta1, theta2, theta3, xy, z, alpha, orientation_tolerance=0., distantce_tolerance=0.1):
        height1 = np.sin(theta1) * self.l1
        height2 = np.sin(theta1 + theta2) * self.l2 + height1
        height3 = np.sin(theta1 + theta2 + theta3) * self.l3 + height2
        width1 = np.cos(theta1) * self.l1
        width2 = np.cos(theta1 + theta2) * self.l2 + width1
        width3 = np.cos(theta1 + theta2 + theta3) * self.l3 + width2
        height = height3
        width = width3

        angle = theta1 + theta2 + theta3
        if height <= z + distantce_tolerance and \
            height >= z - distantce_tolerance and \
            width <= xy + distantce_tolerance and \
            width >= xy - distantce_tolerance and \
            angle <= alpha + orientation_tolerance and \
            angle >= alpha - orientation_tolerance:
            # print("1: ({}, {})\n2: ({}, {})\n3: ({}, {})".format(width1, height1, width2, height2, width3, height3))
            # print("Correct")
            return True
        else:
            # print("1: ({}, {})\n2: ({}, {})\n3: ({}, {})".format(width1, height1, width2, height2, width3, height3))
            # print("Wrong")
            return False

    def checkIfWithinRange(self, theta1, theta2, theta3):
        if theta1 > PI or theta1 < 0 or \
            theta2 > PI/2 or theta2 < -PI/2 or \
            theta3 > PI/2 or theta3 < -PI/2:
            # print("Out theta1: {}\ntheta2: {}\ntheta3: {}".format(theta1,theta2,theta3))
            return False
        else:
            # print("Within theta1: {}\ntheta2: {}\ntheta3: {}".format(theta1,theta2,theta3))
            return True

if __name__ == "__main__":
    arm = RobotArm(0, 10.4, 8.9, 17.5)#, serial_port='/dev/cu.usbserial-1432430')
    isAvailable = arm.move_along((15,0,0),(15,0,30), move=True)
    # isAvailable = arm.move_along_smooth((15,0,0), (15,0,30), 1000, 11, move=True)

    # isAvailable = arm.move_via_smooth([ [20,0,0], [25,0,0], 
    #                                     [30,0,0], [30,5,0], [30,10,0], 
    #                                     [25,10,0], [20,10,0], [20,5,0], 
    #                                     [20,0,0]],
    #  [0, 1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12], True)
    # isAvailable = arm.move_along_smooth_2((15,0,0), (30,0,0), 18, 6, move=False)
    print(isAvailable[0])
    print(isAvailable[1][:, 2:5])
    
    arm.animate_real(isAvailable[0])
    
    fig = plt.figure()
    ax = fig.subplots()
    x = np.array(list(range(len(isAvailable[1][:,0]))))

    ax.plot(x, isAvailable[1][:, 2],label="theta 3")
    ax.plot(x, isAvailable[1][:, 3],label="theta 2")
    ax.plot(x, isAvailable[1][:, 4],label="theta 1")
    ax.plot(x, isAvailable[1][:, 2:5].sum(axis=1),label="sum")

    ax.legend()
    plt.show()
    # isAvailable = arm.move_along((15,0,15),(15,15,15), move=True)
    # arm.animate(isAvailable[1][:, 1:])
    # if arm.checkIfComplete():
    #     val = input("Reset? ")
    #     if val == 'y':
    #         arm.reset()
    #     elif val == 'd':
    #         arm.plot()
    #         arm.reset()

