import numpy as np
import matplotlib.pyplot as plt
from SerialLink import *
import pyfirmata
import time
import serial
from Interpolation import *
from Dijkstra import find_min_path

EPSILON = np.array((1, 1, 1))
        
class RobotArm(SerialLink):
    def __init__(self, serial_port=None):
        DHparams = [
            [96.0355,   0,          0],
            [0,         9.6526,     PI/2], 
            [0,         104.0153,   0],
            [0,         88.6725,    0],
            [0,         170.8423,   0]]
        super(RobotArm, self).__init__(DHparams)
        self.l0, self.l1, self.l2, self.l3 = (9.6526, 104.0153, 88.6725, 170.8423)
        if serial_port is not None:
            self.serial = serial.Serial(serial_port, 9600)
        else:
            self.serial = None
        self.last_pose = [1500,1500,1500,1500,1500,1500]
        self.last_thetas = [np.pi/2, np.pi/2, 0, 0 ,0]#[np.pi/2, np.pi/2, 0, 0 ,0]
        self.forward_kinematics_links_end(self.last_thetas)
    
    def reset(self):
        self.serial.write(str.encode("0:1500;1500;1500;1500;1500;1500;1000|:"))

    ''' Belows are the IK solution '''
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
        theta0 = 0
        if y == 0 and x >= 0:
            theta0 = 0
        elif y == 0 and x < 0:
            theta0 = PI
        else:
            theta0 = np.arctan(x/y)

        xy = np.sqrt(x**2 + y**2)
        x_2 = xy-self.l0 + self.l3 * np.cos(alpha+PI)
        y_2 = z-96.0355 + self.l3 * np.sin(alpha+PI)
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
        if not self.checkSatisfied([theta0, theta1, theta2, theta3, 0], x, y, z, alpha):
            s_2 = np.sqrt(1 - c_2**2)
            theta2 = np.arctan2(s_2, c_2)
            k1 = self.l1 + self.l2 * np.cos(theta2)
            k2 = self.l2 * np.sin(theta2)
            theta1 = np.arctan2(y_2, x_2) - np.arctan2(k2, k1)
            theta3 = alpha - theta1 - theta2
            if not self.checkSatisfied([theta0, theta1, theta2, theta3, 0], x, y, z, alpha):
                # print("Not Satisfied Soltion for ({},{},{}), {}".format(x,y,z,round(alpha/PI*180,2)))
                return None
        
        return (theta0, theta1, theta2, theta3, 0, 0)

    def inverse_kinematics_jacobian_transpose(self, x,y,z):
        thetas = self.last_thetas
        Jt, end_position = self.jacobian(thetas)
        target_position = np.array((x,y,z))
        step = 0.001
        D = 0.01
        thetasLists = []
        count = 0
        lastClampedThetas = np.array(thetas) # Used to check if has reached the local minimal 
        # Check if the current end effector position is close enough
        while (not np.all(np.abs(target_position - end_position) <= EPSILON) or
                not self.checkIfWithinRange_(thetas)[0]):
            
            # Get Descent Orientation
            V = target_position - end_position
            clampedV = V if np.linalg.norm(V) <= D else D * V / np.linalg.norm(V)
            d_thetas = np.dot(Jt , clampedV)
            thetas += d_thetas * step

            # Clamp the thetas within joint limits
            withinRange, clampedThetas = self.checkIfWithinRange_(thetas)
            if not withinRange:
                thetas = clampedThetas.copy()
                # If the target point is not reachable, return the closest solution
                if np.all(np.abs(lastClampedThetas - clampedThetas) <= np.repeat(0.001,5)):
                    count += 1
                    if count == 100:
                        return clampedThetas
                lastClampedThetas = clampedThetas.copy()
            # Get New Jacobian Transpose
            Jt, end_position = self.jacobian(thetas)
            delta = target_position - end_position
            thetasLists.append(thetas.copy())
            print("delta {},{},{}".format(delta[0], delta[1], delta[2]))
            print("thetas [{},{},{},{},{}]".format(thetas[0], thetas[1], thetas[2], thetas[3], thetas[4]))
            print("-------------------------{},{},{}-------------------------".format(x,y,z))
        print("IK solution for {},{},{} is [{},{},{},{},{}]".format(x,y,z,thetas[0], thetas[1], thetas[2], thetas[3], thetas[4]))
        print("===============================================\n\n")
        return thetas

    def inverse_kinematics_damped_least_squares(self, x,y,z):
        thetas = self.last_thetas
        Jt, end_position = self.jacobian(thetas)
        target_position = np.array((x,y,z))
        step = 0.1
        D = 1
        Lambda = 0.1
        thetasLists = []
        count = 0
        # Check if the current end effector position is close enough
        while (not np.all(np.abs(target_position - end_position) <= EPSILON) or
                not self.checkIfWithinRange_(thetas)[0]):
            
            # Get Descent Orientation
            V = target_position - end_position
            clampedV = V if np.linalg.norm(V) <= D else D * V / np.linalg.norm(V)
            J = Jt.T
            JJt = np.dot(J, Jt)
            damped_JJt = JJt + np.diag([Lambda for i in range(JJt.shape[0])])
            damped_JJt = np.linalg.inv(damped_JJt)
            damped_JJt = np.dot(Jt, damped_JJt)
            d_thetas = np.dot(damped_JJt, clampedV)
            thetas += d_thetas * step

            # Clamp the thetas within joint limits
            withinRange, thetas = self.checkIfWithinRange_(thetas)

            # Get New Jacobian Transpose
            Jt, end_position = self.jacobian(thetas)
            delta = target_position - end_position
            thetasLists.append(thetas.copy())
            print("delta {},{},{}".format(delta[0], delta[1], delta[2]))
            print("thetas [{},{},{},{},{}]".format(thetas[0], thetas[1], thetas[2], thetas[3], thetas[4]))
            print("----------------------{}---{},{},{}-------------------------".format(count,x,y,z))
            # if (np.all(np.abs(target_position - end_position) <= EPSILON)):
            #     result, clamped_thetas = self.checkIfWithinRange_(thetas)
            #     if not result:
            #     # if is close enough but violate the restriction, choose another point 
            #         possible_thetas = self.inverse_kinematics(x,y,z)
            #         thetas = possible_thetas[0][0][:5]
            #         print(thetas)
            #         Jt, end_position = self.jacobian(thetas)

        return thetas


    ''' Below are two functions used to send signals and control arm via port '''
    def deg_to_signal(self, thetas):
        '''
            change degrees to signals. The input thetas should be the list of thetas
            in the order from top to bottom. 
            Add time interval based on the min joint difference between last pose
        '''
        print(thetas)
        result = ""
        addon = [PI/2, PI/2, PI/2, PI/2, PI, PI]
        min_theta = PI/2
        for i in range(len(thetas)):
            theta = thetas[i]
            if i == 4 or i == 5:
                signal = int((2000 * (addon[i]-theta) / PI + 500))
            else:
                signal = int((2000 * (addon[i]+theta) / PI + 500))
            diff_theta = np.abs(theta - self.last_pose[i])
            if diff_theta != 0:
                min_theta = diff_theta if diff_theta < min_theta else min_theta
            result += str(signal) + ";"
        min_theta = (min_theta / PI * 2000 + 500) * 0.8
        if min_theta < 20:
            min_theta = 20
        result += str(int(min_theta)) + "|"
        self.last_pose = thetas
        return result

    def signal_to_deg(self, opString):
        '''
            convert signals to degrees.
        '''
        thetaFrames = []
        for opString in opString.split("|")[:-1]:
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
            thetaFrames.append(thetas)
        return thetaFrames

    def send_command(self, opString):
        '''
            send the signals via USB port to control the arm. 
        '''
        # Register the last combination of each joint's degree.
        self.last_thetas = self.signal_to_deg(opString)[-1]
        
        # Send the signal
        if self.serial is not None:
            self.serial.write(str.encode(opString))

    ''' Below are functions to calculate each joint's degrees to the target '''
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

    def move_lerp_jacobian(self, startPos, endPos):
        midPositions = np.linspace(startPos, endPos, num=31)
        thetasLists = []
        for midPosition in midPositions:
            self.last_thetas = self.inverse_kinematics_jacobian_transpose(midPosition[0], midPosition[1], midPosition[2])
            thetasLists.append(self.last_thetas.copy())
        return thetasLists

    def move_lerp_dls(self, startPos, endPos):
        midPositions = np.linspace(startPos, endPos, num=31)
        thetasLists = []
        for midPosition in midPositions:
            self.last_thetas = self.inverse_kinematics_damped_least_squares(midPosition[0], midPosition[1], midPosition[2])
            thetasLists.append(self.last_thetas.copy())
        return thetasLists

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

    ''' Below are the checking functions to ensure IK solutions satisfy the requirements '''
    def checkIfComplete(self):
        line = self.serial.readline(30)
        if line == b'Poses Complete\r\n':
            return True
        else:
            return False

    def checkSatisfied(self, thetas, x, y, z, alpha):
        return self.checkIfWithinRange_(thetas)[0] and \
            self.checkIfCorrect(thetas[1], thetas[2], thetas[3], x,y,z, alpha)

    def checkIfCorrect(self, theta1, theta2, theta3, x, y, z, alpha, orientation_tolerance=0., distantce_tolerance=0.1):
        xy = np.sqrt(x**2 + y**2)
        height1 = np.sin(theta1) * self.l1
        height2 = np.sin(theta1 + theta2) * self.l2 + height1
        height3 = np.sin(theta1 + theta2 + theta3) * self.l3 + height2
        width1 = np.cos(theta1) * self.l1
        width2 = np.cos(theta1 + theta2) * self.l2 + width1
        width3 = np.cos(theta1 + theta2 + theta3) * self.l3 + width2
        height = height3
        width = width3

        angle = theta1 + theta2 + theta3
        if height <= z - 96.0355 + distantce_tolerance and \
            height >= z - 96.0355 - distantce_tolerance and \
            width <= xy - self.l0 + distantce_tolerance and \
            width >= xy - self.l0 - distantce_tolerance and \
            angle <= alpha + orientation_tolerance and \
            angle >= alpha - orientation_tolerance:
            # print("1: ({}, {})\n2: ({}, {})\n3: ({}, {})".format(width1, height1, width2, height2, width3, height3))
            # print("Correct")
            return True
        else:
            # print("1: ({}, {})\n2: ({}, {})\n3: ({}, {})".format(width1, height1, width2, height2, width3, height3))
            # print("Wrong")
            return False

    def checkIfWithinRange(self, theta0, theta1, theta2, theta3):
        Err = 0.01
        if np.abs(theta0%PI - PI) > Err or np.abs(theta0%-PI) > Err:
            print("Out theta0: {}".format(theta0))
            return False
        if np.abs(theta1%PI - PI) > Err or np.abs(theta1%-PI) > Err:
            print("Out theta1: {}".format(theta1))
            return False
        if np.abs(theta2%PI - PI) > Err or np.abs(theta2%-PI) > Err:
            print("Out theta2: {}".format(theta2))
            return False
        if np.abs(theta3%PI - PI) > Err or np.abs(theta3%-PI) > Err:
            print("Out theta3: {}".format(theta3))
            return False
        
        # print("Within theta1: {}\ntheta2: {}\ntheta3: {}".format(theta1,theta2,theta3))
        return True

    def checkIfWithinRange_(self, thetas):
        Err = 0.0
        inrange_thetas = []
        withinRange = np.repeat(False, 5)
        if thetas[0] - PI > Err: # if larger than pi
            # print("Out theta0: {}".format(thetas[0]))
            inrange_thetas.append(PI)
        elif thetas[0] < -Err: # if smaller than 0
            # print("Out theta0: {}".format(thetas[0]))
            inrange_thetas.append(0)
        else:
            inrange_thetas.append(thetas[0])
            withinRange[0] = True

        if thetas[1] - PI > Err:
            # print("Out theta1: {}".format(thetas[1]))
            inrange_thetas.append(PI)
        elif thetas[1] < -Err:
            # print("Out theta1: {}".format(thetas[1]))
            inrange_thetas.append(0)
        else:
            inrange_thetas.append(thetas[1])
            withinRange[1] = True
            
        if thetas[2] - PI/2 > Err:
            # print("Out theta2: {}".format(thetas[2]))
            inrange_thetas.append(PI/2)
        elif thetas[2] + PI/2 < -Err:
            # print("Out theta2: {}".format(thetas[2]))
            inrange_thetas.append(-PI/2)
        else:
            inrange_thetas.append(thetas[2])
            withinRange[2] = True
            
        if thetas[3] - PI/2 > Err:
            # print("Out theta3: {}".format(thetas[3]))
            inrange_thetas.append(PI/2)
        elif thetas[3] + PI/2 < -Err:
            # print("Out theta3: {}".format(thetas[3]))
            inrange_thetas.append(-PI/2)
        else:
            inrange_thetas.append(thetas[3])
            withinRange[3] = True
        
        inrange_thetas.append(thetas[4])
        withinRange[4] = True
        # print("Within theta1: {}\ntheta2: {}\ntheta3: {}".format(theta1,theta2,theta3))
        return np.all(withinRange), np.array(inrange_thetas)

if __name__ == "__main__":
    arm = RobotArm()#, serial_port='/dev/ttyUSB0')
    # thetasLists = arm.inverse_kinematics_jacobian_transpose(10,10,250)
    # thetasLists = arm.move_lerp([150,20,250], [-150,20,250])
    # arm.plot()
    thetasLists = arm.move_lerp_jacobian([150,20,250], [-150,20,250])
    midPoses = []
    for result in thetasLists:
            result = list(reversed(result))
            midPoses.append(result)
    arm.animate(np.array(midPoses))