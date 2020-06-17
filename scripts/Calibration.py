#!/usr/bin/env python

import numpy as np
import cv2
import glob

# Wait time to show calibration in 'ms'
WAIT_TIME = 100

# termination criteria for iterative algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# generalizable checkerboard dimensi`ons
# https://stackoverflow.com/questions/31249037/calibrating-webcam-using-python-and-opencv-error?rq=1
cbrow = 6
cbcol = 8

def calibration():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # IMPORTANT : Object points must be changed to get real physical distance.
    objp = np.zeros((cbrow * cbcol, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)
    print(objp)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('/home/charlie/Project/RobotArm/src/arm_control/scripts/calib_images/*.jpg')
    img = cv2.imread('/home/charlie/Project/RobotArm/src/arm_control/scripts/calib_images/00.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (cbcol, cbrow),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (cbcol, cbrow), corners2, ret)
            cv2.imshow('img',img)
            cv2.waitKey(WAIT_TIME)

    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    if __name__ == "__main__":
        # ---------- Saving the c`alibration -----------------
        cv_file = cv2.FileStorage("/home/charlie/Project/RobotArm/src/arm_control/scripts/calib_images/test.yaml", cv2.FILE_STORAGE_WRITE)
        cv_file.write("camera_matrix", mtx)
        cv_file.write("dist_coeff", dist)

        # note you *release* you don't close() a FileStorage object
        cv_file.release()
    return ret, mtx, dist, rvecs, tvecs

if __name__ == "__main__":
    ret, mtx, dist, rvecs, tvecs = calibration()
    img = cv2.imread('/home/charlie/Project/RobotArm/src/arm_control/scripts/calib_images/00.jpg')

    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h))
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('calibresult.png',dst)
    cv2.imshow('calibresult',dst)
    cv2.waitKey(0)