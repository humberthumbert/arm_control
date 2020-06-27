import numpy as np
import matplotlib.pyplot as plt
from SerialLink import *
import pyfirmata
import time
import serial
from Interpolation import *
from Dijkstra import find_min_path
l0 = 9.6526
class RobotArm(SerialLink):
    def __init__(self, l0, l1, l2, l3, serial_port=None):
        #  104.0153, 88.6725, 170.8423
        DHparams = [
            [0,         0,          0],
            [0,         l0,         PI/2], 
            [0,         l1,         0],
            [0,         l2,         0],
            [0,         l3,         0]]
        super(RobotArm, self).__init__(DHparams)
        self.l0, self.l1, self.l2, self.l3 = (l0, l1, l2, l3)
        if serial_port is not None:
            self.serial = serial.Serial(serial_port, 9600)
        else:
            self.serial = None
        self.lastPose = [1500,1500,1500,1500,1500,1500]

    def reset(self):
        self.serial.write(str.encode("0:1500;1500;1500;1500;1500;1500;1000|:"))

    def inverse_kinematics_frames(self, midPositions):        
        '''
        inverse_kinematics_frames get all the possible combinations for each positions
        and then use Dijkstra algorithm to find the minimal joints' changes among each
        frame.
        '''
        thetalists_frames = []
        alphalists_frames = []
        for midPos in midPositions:
            thetalists, alphalists = self.inverse_kinematics(midPos[0], midPos[1], midPos[2])
            if len(thetalists) == 0:
                print("Position {},{},{} is unreachable".format(midPos[0], midPos[1], midPos[2]))
                return
            thetalists_frames.append(np.array(thetalists))
            print("frame {}: length {}".format(len(thetalists_frames), len(thetalists)))
            alphalists_frames.append(alphalists)

        results = find_min_path(thetalists_frames)
        return results

    def inverse_kinematics(self, x, y, z): 
        '''
        inverse_kinematics tries all the possible degrees of the end effector
        and returns possible combinations of joints and corresponding degrees
        '''
        thetas = None
        thetas_list = []
        alpha_list = []
        for time in range(2):
            for alpha in np.linspace(-PI, PI, num=180*np.power(10,time)+1):
                thetas = self.inverse_kinematics_(x, y, z, alpha)
                if thetas is not None:
                    thetas_list.append(list(thetas))
                    alpha_list.append(alpha)
        return (thetas_list, alpha_list)


    def inverse_kinematics_(self, x, y, z, alpha):
        '''
        inverse_kinematics_ solve the ik solution for desired x,y,z and alpha
        for the end effector. Return the elbow up solution if possible,
        then elbow down solutions, otherwise None
        '''
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
        theta0 = 0
        if y == 0 and x >= 0:
            theta0 = PI/2
        elif y == 0 and x < 0:
            theta0 = -PI/2
        else:
            theta0 = np.arctan(x/y)
        
        print(theta0 / np.pi * 180)
        return (theta0, theta1, theta2, theta3, 0, 0)

    def inverse_kinematics__(self, x,y,z, alpha):
        return

    def deg_to_signal(self, thetas):
        '''
        change degrees to signals. The input thetas should be the list of thetas
        in the order from top to bottom. 
        Add time interval based on the min joint difference between last pose
        '''
        print(thetas)
        result = ""
        addon = [PI/2, PI/2, PI/2, PI/2, PI, PI/2]
        min_theta = PI/2
        for i in range(len(thetas)):
            theta = thetas[i]
            if i == 4:
                signal = int((2000 * (addon[i]-theta) / PI + 500))
            else:
                signal = int((2000 * (addon[i]+theta) / PI + 500))
            diff_theta = np.abs(theta - self.lastPose[i])
            if diff_theta != 0:
                min_theta = diff_theta if diff_theta < min_theta else min_theta
            result += str(signal) + ";"
        min_theta = (min_theta / PI * 2000 + 500) * 0.8
        if min_theta < 20:
            min_theta = 20
        result += str(int(min_theta)) + "|"
        self.lastPose = thetas
        return result

    def send_command(self, opString):
        if self.serial is not None:
            self.serial.write(str.encode(opString))

    def move_to(self, x, y, z):
        result = self.inverse_kinematics(x,y,z)
        if result is not None:
            result = list(reversed(result[0][0]))  # use the smallest alpha
            print("\n\n\n")
            print(result)
            thetas = "0:" + self.deg_to_signal(result) + ":"
            return (thetas, np.array(result))
        else:
            print("Cannot move to target position")
            return None

    def move_slerp(self, startPos, endPos):
        midPositions = np.array([slerp(np.array(startPos), np.array(endPos), t) for t in np.arange(0.0, 1.0, 0.05)])
        results = self.inverse_kinematics_frames(midPositions)
        thetas = ""
        index = 0
        midPoses = []
        for result in results:
            result = list(reversed(result))
            midPoses.append(result)
            result = self.deg_to_signal(result)
            thetas += str(index) + ":" + result
            index += 1
        
        return (thetas + ":", np.array(midPoses))

    def move_lerp(self, startPos, endPos):
        midPositions = np.linspace(startPos, endPos, num=31)
        results = self.inverse_kinematics_frames(midPositions)
        thetas = ""
        index = 0
        midPoses = []
        if results is None:
            print("Positions not reachable")
            return
        for result in results:
            result = list(reversed(result))
            midPoses.append(result)
            result = self.deg_to_signal(result)
            thetas += str(index) + ":" + result
            index += 1

        return (thetas + ":", np.array(midPoses))

    def move_along(self, startPos, endPos):
        midPositions = np.linspace(startPos, endPos, num=21)
        thetas = ""
        index = 0
        midPoses = []
        for midPos in midPositions:
            result = self.move_to(midPos[0], midPos[1], midPos[2])
            if result is None:
                print("Cannot move along this line")
                return None
            else:
                midPoses.append(result)
                result = self.deg_to_signal(result)
                thetas += str(index) + ":" + result
                index += 1
        return (thetas + ":", np.array(midPoses))

    def move_perp(self, startPos, endPos, time):  
        '''
        Pose intERPolation
        '''
        positions = np.array([list(startPos), list(endPos)])
        results = self.inverse_kinematics_frames(positions)
        if (results is None):
            print("Cannot calculate the pose")
            return
        else:
            startThetas, endThetas = results
        
        # theta(t) = a0 + a1 * t + a2 * t^2 + a3 * t^3
        params = np.zeros((len(startThetas),4))
        for j in range(len(startThetas)):
            params[j][0] = startThetas[j]
            params[j][1] = 0
            params[j][2] = 3 * (endThetas[j] - startThetas[j]) / (time**2)
            params[j][3] = -2 * (endThetas[j] - startThetas[j]) / (time**3)
        
        # sample by time
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

        return (thetas + ":", midPoses)

    def move_along_smooth(self, startPos, endPos, time, num):
        midPositions = np.linspace(startPos, endPos, num=num)
        thetas_commands = ""
        midPoses_list = []
        for i in range(len(midPositions)-1):
            result = self.move_perp(midPositions[i], midPositions[i+1], time=time/(num-1))
            if result is not None:
                thetas, midPoses = result
                thetas = thetas[:-1]
                thetas_commands += thetas
                midPoses_list.append(midPoses.tolist())

        midPoses = np.array(midPoses_list)
        midPoses = midPoses.reshape(-1, 6)
        return (thetas_commands + ":", midPoses)

    def move_along_smooth_2(self, startPos, endPos, time, num):
        midPositions = np.linspace(startPos, endPos, num=num)
        times = np.linspace(0, time, num=num)
        return self.move_via_smooth(midPositions, times)

    def move_via_smooth(self, positions, times):
        thetas = []
        for i in range(len(positions)):
            result = self.move_to(positions[i][0], positions[i][1], positions[i][2])
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
    arm = RobotArm(0, 104.0153, 88.6725, 170.8423, serial_port='/dev/ttyUSB0')
    isAvailable = arm.move_to(150,200.0,0)
    arm.send_command(isAvailable[0])
    print(isAvailable[0])
    # arm.animate_real(isAvailable[0])

    # fig = plt.figure()
    # ax = fig.subplots()
    # x = np.array(list(range(len(isAvailable[1][:,0]))))
    # ax.plot(x, isAvailable[1][:, 2],label="theta 3")
    # ax.plot(x, isAvailable[1][:, 3],label="theta 2")
    # ax.plot(x, isAvailable[1][:, 4],label="theta 1")
    # ax.plot(x, isAvailable[1][:, 2:5].sum(axis=1),label="sum")
    # ax.legend()
    # plt.show()

    # isAvailable = arm.move_along((15,0,15),(15,15,15), move=True)
    # arm.animate(isAvailable[1][:, 1:])
    # if arm.checkIfComplete():
    #     val = input("Reset? ")
    #     if val == 'y':
    #         arm.reset()
    #     elif val == 'd':
    #         arm.plot()
    #         arm.reset()


    # isAvailable = arm.move_lerp((10,10,0),(10,30,0))
    # arm.send_command(isAvailable[0])
    # arm.animate_real(isAvailable[0])

    # isAvailable = arm.move_lerp((10,30,0),(-10,30,0))
    # arm.send_command(isAvailable[0])
    # arm.animate_real(isAvailable[0])

    # isAvailable = arm.move_lerp((-10,30,0),(-10,10,0))
    # arm.send_command(isAvailable[0])
    # arm.animate_real(isAvailable[0])

    # isAvailable = arm.move_slerp((-10,10,0),(10,10,0))
    # arm.send_command(isAvailable[0])
    # arm.animate_real(isAvailable[0])

    # For later
        #  104.0153, 88.6725, 170.8423
        # DHparams = [
        #     [0,         0,          0],
        #     [96.0355,   9.6526,     PI/2], 
        #     [0,         104.0153,   0],
        #     [0,         88.6725,    0],
        #     [0,         170.8423,   0]]
        # super(RobotArm, self).__init__(DHparams)
        # self.l0, self.l1, self.l2, self.l3 = (9.6526, 104.0153, 88.6725, 170.8423)