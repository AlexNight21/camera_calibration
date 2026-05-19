import cv2
import os
import numpy as np
from pathlib import Path

# params
boardSize = (9, 6)
frSize = (1600, 1200)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# data path
imgs_path = os.path.join(Path(__file__).parents[1], "data")


def make_cam_calibration(imgs_path, board_size, frame_size, criteria):
    
    objPoints = []    # 3D points  
    imgPoints = []    # 2D points plane
    
    objp = np.zeros((1, board_size[0] * board_size[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    
    for img_name in os.listdir(imgs_path):
        if not img_name.endswith((".jpg", ".png", ".jpeg")):
            continue
        
        cur_img_path = os.path.join(imgs_path, img_name)
        
        image = cv2.imread(cur_img_path)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # find board corners
        ret, corners = cv2.findChessboardCorners(
            image, 
            board_size, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        
        if ret:
            objPoints.append(objp)
            
            corners2 = cv2.cornerSubPix(
                image_gray, 
                corners, 
                (11,11),
                (-1,-1), 
                criteria,
            )
            
            imgPoints.append(corners2)
        
    print(f"[INFO] num of valid images: {len(imgPoints)}")
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objPoints, 
        imgPoints, 
        frSize, 
        None, 
        None,
    )
    
    print(
        f"cam calibrated: {ret}\n"
        f"cam matrix:\n{mtx}\n"
        f"distortion coeffs:\n{dist}\n"
        f"rotation vectors:\n{rvecs}\n"
        f"translation vectors:\n{tvecs}\n"
    )

if __name__ == "__main__":
    make_cam_calibration(imgs_path, boardSize, frSize, criteria)