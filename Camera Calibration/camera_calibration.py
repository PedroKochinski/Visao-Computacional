#!/usr/bin/env python
 
import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt



# Defining the dimensions of checkerboard
CHECKERBOARD = (7,7)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = [] 
 
 
# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None
 
# Extracting path of individual image stored in a given directory
images = glob.glob('./fotos/*.jpg')


for fname in images:
    img = cv2.imread(fname)
    plt.imshow(img)
    plt.show()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    lwr = np.array([0,0,90])
    upr = np.array([179, 61, 252])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    msk = cv2.inRange(hsv, lwr, upr)
    krn = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 30))
    dlt = cv2.dilate(msk, krn, iterations=5)
    res = 255 - cv2.bitwise_and(dlt, msk)
    res = np.uint8(res)
    res2 = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    plt.imshow(res2)
    plt.show()

    ret, corners = cv2.findChessboardCorners(res, (7, 7),
                                            flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                cv2.CALIB_CB_FAST_CHECK +
                                                cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_EXHAUSTIVE)
    if ret:
        fnl = cv2.drawChessboardCorners(img, (7, 7), corners, ret)
        img = cv2.cvtColor(fnl, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show()

    else:
        print("No Checkerboard Found")

    if ret:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)

        imgpoints.append(corners2)

        # Draw and display the corners
        img_refined = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

    img_refined = cv2.cvtColor(img_refined, cv2.COLOR_BGR2RGB)
    plt.imshow(img_refined)
    plt.show()

    h,w = img.shape[:2]
 
"""
Realizando a calibração da camera passando os valores dos pontos 3D conhecidos (objpoints)
e as coordenadas de pixel correspondentes dos cantos detectados (imgpoints)
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
 
print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)

"""
Vamos remover agora a distorção sob outra foto tirada pela mesma camera
"""
img = cv2.imread('./foto_ex.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
img_final = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
plt.imshow(img_final)
plt.show()
cv2.imwrite('calibresult.png', img_final)
