import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import time

PI = np.pi

def slerp(p0, p1, t):
        omega = np.arccos(np.dot(p0/norm(p0), p1/norm(p1)))
        so = np.sin(omega)
        return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1


def move_via_smooth_fixed_accelerate(thetas, positions, times):
    # params = np.zeros((len(startThetas), 4, ))
    accelerate = 50 / 180 * PI # deg/s^2
    thetas = np.array(thetas).T
    print("\n-------------Start Interpolation Fixed Acceleration--------------\n")
    print(thetas)
    results = []
    for s in range(len(thetas)): # joint id
        print("\nIndex: {}".format(s))
        print(thetas[s])
        parabolic_intervals = []
        linear_intervals = []
        linear_velocities = []
        accelerates = []

        # First path planning
        theta_0 = thetas[s][0]
        theta_1 = thetas[s][1]
        time = times[1]
        sign = np.sign(theta_1 - theta_0)
        sign = 1 if sign == 0 else sign
        dd_theta_0 = sign * np.abs(accelerate)
        print("Theta_0: {}, \tTheta_1: {}, \tTime: {}, \tAccelerate: {}".format(theta_0, theta_1, time, dd_theta_0))
        t_0 = time - np.sqrt(np.power(time, 2) - 2 * (theta_1 - theta_0) / dd_theta_0)
        d_theta_01 = (theta_1 - theta_0) / (time - t_0 / 2)
        parabolic_intervals.append(t_0)
        linear_velocities.append(d_theta_01)
        accelerates.append(dd_theta_0)
        print("t_1: {}, v_1: {}, a_1: {}".format(t_0, d_theta_01, dd_theta_0))

        # Middle path planning
        for t in range(2, len(thetas[s])-1): # position id
            theta_k = thetas[s][t]
            theta_j = thetas[s][t-1]
            theta_l = thetas[s][t+1]
            time_jk = times[t] - times[t-1]
            time_kl = times[t+1] - times[t]
            d_theta_jk = (theta_k - theta_j) / time_jk
            d_theta_kl = (theta_l - theta_k) / time_kl
            sign = np.sign(d_theta_kl - d_theta_jk)
            sign = 1 if sign == 0 else sign
            dd_theta_k = sign * np.abs(accelerate)
            t_k = np.abs(d_theta_kl - d_theta_jk) / accelerate
            parabolic_intervals.append(t_k)
            linear_velocities.append(d_theta_jk)
            accelerates.append(dd_theta_k)
            print("t_{}: {}, v_{}: {}, a_{}: {}".format(t, t_k, t, d_theta_jk, t, dd_theta_k))

        # Last path planning
        n = len(thetas[s]) - 1
        theta_n_1 = thetas[s][n-1]
        theta_n = thetas[s][n]
        time_nn1 = times[n]
        sign = np.sign(theta_n - theta_n_1)
        sign = 1 if sign == 0 else sign
        dd_theta_n = sign * np.abs(accelerate)
        t_n = time_nn1 - np.sqrt(np.power(time_nn1, 2) - \
            2 * (theta_n - theta_n_1) / dd_theta_n)
        d_theta_n1n = (theta_n - theta_n_1) / (time_nn1 - t_n / 2)
        parabolic_intervals.append(t_n)
        linear_velocities.append(d_theta_n1n)
        accelerates.append(dd_theta_n)
        print("t_{}: {}, v_{}: {}, a_{}: {}".format(n, t_n, n, d_theta_n1n, n, dd_theta_n))
        
        # linear interval
        t_01 = times[1] - parabolic_intervals[0] - parabolic_intervals[1] / 2
        linear_intervals.append(t_01)
        for t in range(2, len(parabolic_intervals)):
            t_jk = (times[t] - times[t-1]) - parabolic_intervals[t-1] / 2 - parabolic_intervals[t] / 2
            linear_intervals.append(t_jk)
        t_n1n = times[n] - parabolic_intervals[len(parabolic_intervals) - 1] - \
            parabolic_intervals[len(parabolic_intervals) - 2] / 2
        linear_intervals.append(t_n1n)
        
        # timeline
        print("linear interval:")
        print(linear_intervals)

        print("parabolic_interval: ")
        print(parabolic_intervals)
        
        print("linear velocities: ")
        print(linear_velocities)
        timeline = []
        time_finish = 0
        for i in range(len(linear_intervals)):
            time_finish += parabolic_intervals[i]
            timeline.append(time_finish)
            time_finish += linear_intervals[i]
            timeline.append(time_finish)
        timeline = np.array(timeline)
        print(timeline)
        # calculate thetas
        result = []
        moments = np.linspace(times[0], times[n], num=21)
        for moment in moments:
            index = np.searchsorted(timeline, moment)
            print(index)
            if index % 2 == 0:
                i = int(index / 2)
                theta = thetas[s][i] + linear_velocities[i] * (moment - times[i]) + \
                    accelerates[i]/2 * (moment - times[i+1] + linear_intervals[i] / 2)**2
            else:
                i = int(index / 2)
                theta = thetas[s][i] + linear_velocities[i] * (moment - times[i])
            result.append(theta)
        results.append(result)
    return results

def move_via_smooth_fixed_time(thetas, positions, times):
    accel_time = 0.25 # second
    thetas = np.array(thetas).T
    print("\n-------------Start Interpolation Fixed Time--------------\n")
    print(thetas)
    results = []
    for s in range(len(thetas)): # joint id
        print("===============Index {}===============".format(s))
        print(thetas[s])
        parabolic_intervals = []
        linear_intervals = []
        linear_velocities = []
        accelerates = []

        # First path planning
        theta_0 = thetas[s][0]
        theta_1 = thetas[s][1]
        time = times[1] - times[0]
        d_theta_01 = (theta_1 - theta_0) / (time - accel_time / 2)
        dd_theta_0 = d_theta_01 / accel_time
        t_01 = time - 3 * accel_time / 2
        # print("Theta_0: {}, \tTheta_1: {}, \tTime: {}, \tAccelerate: {}".format(theta_0, theta_1, time, dd_theta_0))
        print("t_01: {}".format(t_01))
        linear_intervals.append(t_01)
        linear_velocities.append(d_theta_01)
        accelerates.append(dd_theta_0)

        # Middle path planning
        for t in range(1, len(thetas[s])-1): # position id
            theta_k = thetas[s][t]
            theta_j = thetas[s][t-1]
            theta_l = thetas[s][t+1]
            time_jk = times[t] - times[t-1]
            time_kl = times[t+1] - times[t]
            d_theta_jk = (theta_k - theta_j) / time_jk
            d_theta_kl = (theta_l - theta_k) / time_kl
            dd_theta_k = (d_theta_kl - d_theta_jk) / accel_time
            t_jk = time_jk - accel_time
            print("t_{}{}: {}".format(t-1, t, t_jk))
            if t != 1:
                linear_intervals.append(t_jk)
                linear_velocities.append(d_theta_jk)
            accelerates.append(dd_theta_k)
            # print("t_{}: {}, v_{}: {}, a_{}: {}".format(t, t_jk, t, d_theta_jk, t, dd_theta_k))

        # Last path planning
        n = len(thetas[s]) - 1
        theta_n_1 = thetas[s][n-1]
        theta_n = thetas[s][n]
        time_nn1 = times[n] - times[n-1]
        d_theta_n1n = (theta_n - theta_n_1) / (time_nn1 - accel_time / 2)
        dd_theta_n = d_theta_n1n / -1 / accel_time # TODO why have to add negative sign
        t_n = time_nn1 - 3 * accel_time / 2
        linear_intervals.append(t_n)
        linear_velocities.append(d_theta_n1n)
        accelerates.append(dd_theta_n)
        #  print("t_{}: {}, v_{}: {}, a_{}: {}".format(n, t_n, n, d_theta_n1n, n, dd_theta_n))

        # parabolic interval
        for t in range(len(linear_intervals)):
            parabolic_intervals.append(accel_time)
        print(linear_intervals)
        # timeline
        timeline = []
        time_finish = 0
        for i in range(len(linear_intervals)):
            time_finish += parabolic_intervals[i]
            timeline.append(time_finish)
            time_finish += linear_intervals[i]
            timeline.append(time_finish)
        timeline = np.array(timeline)
        print(timeline)
        
        # calculate thetas
        result = []
        times_copy = np.copy(times)
        times_copy[0] = times_copy[0] + accel_time / 2.
        times_copy[n] = times_copy[n] - accel_time / 2.
        num = (times[n] - times[0]) * 10 + 1
        moments = np.linspace(times[0], times[n], num=num)
        for moment in moments:
            index = np.searchsorted(timeline, moment)
            if index % 2 == 0:
                i = int(index / 2)
                if i == 0:
                    theta = thetas[s][i] + accelerates[i]/2 * moment**2
                else:
                    theta = thetas[s][i-1] + linear_velocities[i-1] * (moment - times_copy[i-1]) + \
                        accelerates[i-1]/2 * (moment - timeline[index-1] + parabolic_intervals[i-1] / 2)**2
            else:
                i = int(index / 2)
                theta = thetas[s][i] + linear_velocities[i] * (moment - times_copy[i])
            result.append(theta)
        results.append(result)
    return results



# test code
if __name__ == '__main__':
    pA = np.array([5.0, 0.0, -5.0])
    pB = np.array([5.0, 0.0, 5.0])
    pC = np.array([-5.0, 0.0, 5.0])
    pD = np.array([-5.0, 0.0, -5.0])

    ps1 = np.array([slerp(pA, pB, t) for t in np.arange(0.0, 1.0, 0.01)])
    ps2 = np.array([slerp(pB, pC, t) for t in np.arange(0.0, 1.0, 0.01)])
    ps3 = np.array([slerp(pC, pD, t) for t in np.arange(0.0, 1.0, 0.01)])
    ps4 = np.array([slerp(pA, pD, t) for t in np.arange(0.0, 1.0, 0.01)])
    from pylab import *
    from mpl_toolkits.mplot3d import Axes3D
    f = figure()
    ax = Axes3D(f)
    axis_min = ps1.min()
    axis_max = ps1.max()
    ax.set_xlim3d(axis_min, axis_max)
    ax.set_ylim3d(axis_min, axis_max)
    ax.set_zlim3d(axis_min, axis_max)
    ax.plot3D(ps1[:,0], ps1[:,1], ps1[:,2], '.')
    ax.plot3D(ps2[:,0], ps2[:,1], ps2[:,2], '.')
    ax.plot3D(ps3[:,0], ps3[:,1], ps3[:,2], '.')
    ax.plot3D(ps4[:,0], ps4[:,1], ps4[:,2], '.')
    show()