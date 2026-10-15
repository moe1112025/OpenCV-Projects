"""Undistort an image using calibration data or clearly labelled demo coefficients."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
HERE=Path(__file__).resolve().parent
DEFAULT_IMAGE=HERE/'test_images'/'lens_grid.png'

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--image',type=Path,default=DEFAULT_IMAGE); ap.add_argument('--calibration',type=Path,help='NPZ created by camera_calibration.py'); ap.add_argument('--output',type=Path,default=HERE/'outputs'/'undistorted.png'); ap.add_argument('--alpha',type=float,default=1.0,help='0 crops more; 1 retains more pixels'); ap.add_argument('--show',action='store_true'); args=ap.parse_args()
    if not 0<=args.alpha<=1: raise SystemExit('--alpha must be between 0 and 1')
    image=cv2.imread(str(args.image))
    if image is None: raise SystemExit(f'ERROR: Could not read input: {args.image}')
    h,w=image.shape[:2]
    if args.calibration:
        if not args.calibration.is_file(): raise SystemExit(f'ERROR: Calibration file not found: {args.calibration}')
        with np.load(args.calibration) as data:
            camera_matrix=data['camera_matrix']; dist_coeffs=data['dist_coeffs']
        print('[INFO] Using provided calibration data.')
    else:
        # Demonstration only: never use these coefficients for scientific measurements.
        camera_matrix=np.array([[w,0,w/2],[0,w,h/2],[0,0,1]],np.float64)
        dist_coeffs=np.array([-.16,.035,0,0,0],np.float64)
        print('[WARN] Using fabricated demo coefficients. For real images, use --calibration from your own camera.')
    new_matrix,roi=cv2.getOptimalNewCameraMatrix(camera_matrix,dist_coeffs,(w,h),args.alpha,(w,h))
    output=cv2.undistort(image,camera_matrix,dist_coeffs,None,new_matrix)
    x,y,rw,rh=roi
    if rw>0 and rh>0: cv2.rectangle(output,(x,y),(x+rw,y+rh),(0,220,255),1)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(args.output),output): raise SystemExit('ERROR: Could not save result.')
    print(f'[OK] Saved: {args.output.resolve()}')
    if args.show: cv2.imshow('Input',image); cv2.imshow('Undistorted',output); cv2.waitKey(0); cv2.destroyAllWindows()
if __name__=='__main__': main()
