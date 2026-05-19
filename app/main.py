import cv2
import os
import pickle
import numpy as np
from pathlib import Path


# params
boardSize = (9, 6)
frSize = (1600, 1200)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# data path
imgs_path = os.path.join(Path(__file__).parents[1], "data/calibr_dataset")
result_path = os.path.join(Path(__file__).parents[1], "data/result_imgs")
calibr_file = os.path.join(Path(__file__).parents[1], "data/calibration.pckl")

# test
test_img_name = "WIN_20260519_15_10_43_Pro.jpg"


def save_calibration_results(calibr_file, mtx, opt_mtx, dist):
    '''save calibration results to file'''
    
    calibr_results = {}
    calibr_results["mtx"] = mtx
    calibr_results["opt_mtx"] = opt_mtx
    calibr_results["dist"] = dist
    
    with open(calibr_file, "wb") as f:
        pickle.dump(calibr_results, f)

def save_undistorted_imgs(
    imgs_path, 
    img_name, 
    result_path,
    calibr_file,
):
    '''save undistorted image after camera calibration'''
    
    # remove old result images
    for res_img_name in os.listdir(result_path):
        if res_img_name.endswith((".jpg", ".png", ".jpeg")):
            os.remove(os.path.join(result_path, res_img_name))
            print(f"[INFO] removed old result image: {res_img_name}")
    
    with open(calibr_file, "rb") as f:
        calibr_results = pickle.load(f)
        
    mtx = calibr_results["mtx"]
    opt_mtx = calibr_results["opt_mtx"]
    dist = calibr_results["dist"]
    
    img_path = os.path.join(imgs_path, img_name)
    image = cv2.imread(img_path)
    
    image_undistorted = cv2.undistort(
        image, mtx, dist, None, opt_mtx,
    )
    
    # save images
    cv2.imwrite(os.path.join(result_path, f"{Path(img_name).stem}_undistorted" + ".jpg"), image_undistorted)
    cv2.imwrite(os.path.join(result_path, f"{Path(img_name).stem}_original" + ".jpg"), image)
    

def show_corners(image, board_size, corners, ret):
    '''show corners of chessboard images'''
    
    image = cv2.drawChessboardCorners(
        image, 
        board_size, 
        corners, 
        ret,
    )

    cv2.imshow('chessboard_corners', image)
    cv2.waitKey(1000)

def make_cam_calibration(
    imgs_path, 
    board_size, 
    frame_size, 
    criteria, 
    calibr_file,
    show_corners=False,
    save_calibr_data=True,
    save_example=True,
):
    
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
            
            if show_corners:
                show_corners(
                    image=image, 
                    board_size=board_size, 
                    corners=corners2, 
                    ret=ret,
                )
        
    print(f"[INFO] num of valid images: {len(imgPoints)}")
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objPoints, 
        imgPoints, 
        frame_size, 
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
    
    # Returns optimal camera matrix and a rectangular region of interest
    optCamMatrix, _ = cv2.getOptimalNewCameraMatrix(
        mtx, dist, frame_size, 1, frame_size,
    )
    
    if save_calibr_data:
        save_calibration_results(
            calibr_file=calibr_file,
            mtx=mtx,
            opt_mtx=optCamMatrix,
            dist=dist,
        )

if __name__ == "__main__":
    # make_cam_calibration(
    #     imgs_path=imgs_path,
    #     board_size=boardSize, 
    #     frame_size=frSize, 
    #     criteria=criteria,
    #     calibr_file=calibr_file,
    #     show_corners=False,
    #     save_calibr_data=True,
    #     save_example=True,
    # )
    
    save_undistorted_imgs(
        imgs_path=imgs_path,
        img_name=test_img_name,
        result_path=result_path,
        calibr_file=calibr_file,
    )