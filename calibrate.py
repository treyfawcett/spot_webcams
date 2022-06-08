import logging
from os import path
from typing import Optional
import numpy as np
import cv2
import glob
import yaml
# termination criteria

logging.basicConfig(level='INFO')
CHECKERBOARD_DIMS=(6,9)

def camera_gen(capture_dev):
    while True:
        success, img = capture_dev.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if not success:
            raise GeneratorExit
        yield gray
    
def imgdir_gen(img_glob: str):
    img_files = glob.glob(img_glob)
    for fname in img_files:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        yield gray
        
class Calibrator:
    def __init__(self, 
                    data_source:str, 
                    width:int, 
                    height:int, 
                    output_file_name:Optional[str]=None, 
                    draw_visuals:bool=True, 
                    save_images:bool=False) -> None:
        
        self.draw_visuals = draw_visuals
        self.save_images = save_images
        
        if path.isdir(data_source):
            img_glob = path.join(data_source,'capture*.png')
            shapes = [img.shape[::-1] for img in imgdir_gen(img_glob)]
            print(shapes)
            assert len(shapes), f'coud not find any images matching glob {img_glob}'
            
            assert all([s == shapes[0] for s in shapes]), f'irregular image resolutions detected'
            self.width = shapes[0][0]
            self.height = shapes[0][1]
            self.source = imgdir_gen(img_glob)
            self.save_images = False
            
        else:
            try:
                data_source = int(data_source)
            except:
                pass
            self.capture = cv2.VideoCapture(data_source)
            self.source = camera_gen(self.capture)
            if not self.capture.isOpened():
                # Unable to open a video capture connection to the specified device.
                err = "Unable to open a cv2.VideoCapture connection to %s" % data_source
                logging.warning(err)
                raise Exception(err)

            if width > 0 and height > 0:
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logging.info(f'Capture has resolution: {self.width} x {self.height}')

        if output_file_name and (len(output_file_name) < 5 or output_file_name[-5:] != '.yaml'):
            output_file_name+='.yaml'
        self.save_name = output_file_name


    def run(self, img_count):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((CHECKERBOARD_DIMS[0]*CHECKERBOARD_DIMS[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:CHECKERBOARD_DIMS[0],0:CHECKERBOARD_DIMS[1]].T.reshape(-1,2)
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        
        capture_count=0
        gray = None
        for gray in self.source:
            # Find the chess board corners
            logging.debug("iterate")
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_DIMS, None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                capture_count+=1
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                logging.debug(corners2)
                imgpoints.append(corners2)
                # Draw and display the corners
                if self.save_images:
                    cv2.imwrite(f'capture{capture_count:03}.png', gray, [int(cv2.IMWRITE_PNG_COMPRESSION),0])
                if self.draw_visuals:
                    cv2.drawChessboardCorners(gray, CHECKERBOARD_DIMS, corners2, ret)
                    cv2.imshow('capture', gray)
                    cv2.waitKey(100)
                
                if self._check_coverage(imgpoints, (self.width, self.height)):
                    break
                
                if capture_count >= img_count:
                    break

        logging.info(f'starting calibration with {capture_count} images')
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        logging.info(f'camera matrix:')
        logging.info(f'{mtx}')
        logging.info(f'distortion:')
        logging.info(f'{dist}')

        w_h = (self.width, self.height)
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, w_h, 0, w_h)
        logging.info(f'camera matrix (after undistortion):')
        logging.info(f'{newcameramtx}')

        # undistort
        undistorted = cv2.undistort(gray, mtx, dist, None, newcameramtx)
        params={}
        params['width'] = self.width
        params['height'] = self.height
        params['camera_matrix'] = mtx.tolist()
        params['distortion_coefficients'] = dist.tolist()

        if self.save_name:
            with open(f'{self.save_name}','w') as file:
                yaml.dump(params, file)
        
        if self.save_images:
            cv2.imwrite(f'undistorted.jpg', undistorted)
        
        if self.draw_visuals:
            cv2.imshow('undistorted', undistorted)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
            
    def _check_coverage(self, imgpoints, img_size):
        print(len(imgpoints))
        print(imgpoints[-1].shape)

        for point_set in imgpoints:
            for point in point_set:
                x= point[0][0]
                y=point[0][1]
                print((x,y))
        return False
        

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-s', '--source', required=True, default=0, help=('Image source to query.'))
    parser.add_argument('-w', '--res-width', required=False, type=int, default=-1, help="Resolution width (pixels).")
    parser.add_argument('-h', '--res-height', required=False, type=int, default=-1, help="Resolution height (pixels).")

    parser.add_argument('-o', '--output_file_name', required=False, type=str, default="", help="Name of yaml file contianing the results.")

    parser.add_argument('-v', '--draw-visuals', action="store_true", help="Render images to screen.")
    parser.add_argument('-i', '--save-images', action="store_true", help="Save images to disk.")

    options = parser.parse_args()
    c = Calibrator(data_source=options.source, 
                    width=options.res_width, 
                    height=options.res_height, 
                    output_file_name=options.output_file_name,
                    draw_visuals=options.draw_visuals,
                    save_images=options.save_images)
    c.run(30)