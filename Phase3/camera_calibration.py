"""Calibrate a camera from images of a checkerboard with OpenCV."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
HERE=Path(__file__).resolve().parent
DEFAULT_IMAGES=HERE/'test_images'/'calibration_samples'

def parse_pattern(value):
    try:
        w,h=(int(v) for v in value.lower().split('x',1))
        if w<2 or h<2: raise ValueError
        return w,h
    except Exception as exc: raise argparse.ArgumentTypeError('Pattern must be inner corners like 9x6.') from exc

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--images',type=Path,default=DEFAULT_IMAGES,help='Folder with checkerboard photographs')
    ap.add_argument('--pattern',type=parse_pattern,default=(9,6),help='Inner-corner count, e.g. 9x6')
    ap.add_argument('--square-size',type=float,default=1.0,help='Physical square size in chosen units')
    ap.add_argument('--output',type=Path,default=HERE/'outputs'/'demo_only_calibration.npz')
    ap.add_argument('--show-corners',action='store_true')
    args=ap.parse_args()
    if args.square_size<=0: raise SystemExit('ERROR: --square-size must be positive.')
    paths=sorted(p for p in args.images.glob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp'} )
    if not paths: raise SystemExit(f'ERROR: No images found in {args.images}')
    pattern=args.pattern; objp=np.zeros((pattern[0]*pattern[1],3),np.float32)
    objp[:,:2]=np.mgrid[0:pattern[0],0:pattern[1]].T.reshape(-1,2)*args.square_size
    object_points=[]; image_points=[]; image_size=None; accepted=[]
    flags=cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE
    for path in paths:
        image=cv2.imread(str(path))
        if image is None: print(f'[WARN] Skipped unreadable image: {path.name}'); continue
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image_size=(gray.shape[1],gray.shape[0]) if image_size is None else image_size
        if (gray.shape[1],gray.shape[0])!=image_size:
            print(f'[WARN] Skipped image with mismatched resolution: {path.name}'); continue
        found,corners=cv2.findChessboardCorners(gray,pattern,flags)
        if not found and hasattr(cv2,'findChessboardCornersSB'):
            found,corners=cv2.findChessboardCornersSB(gray,pattern)
        if not found or corners is None:
            print(f'[WARN] Checkerboard not detected: {path.name}'); continue
        if len(corners)!=len(objp): print(f'[WARN] Wrong corner count: {path.name}'); continue
        corners=cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,.001))
        object_points.append(objp.copy()); image_points.append(corners); accepted.append(path.name)
        print(f'[OK] Detected {len(corners)} corners in {path.name}')
        if args.show_corners:
            preview=image.copy(); cv2.drawChessboardCorners(preview,pattern,corners,True); cv2.imshow('Checkerboard Corners',preview); cv2.waitKey(250)
    if args.show_corners: cv2.destroyAllWindows()
    if len(accepted)<3: raise SystemExit(f'ERROR: Need at least 3 valid views; found {len(accepted)}. Take 10–20 varied real camera photos for meaningful calibration.')
    rms,camera_matrix,dist_coeffs,rvecs,tvecs=cv2.calibrateCamera(object_points,image_points,image_size,None,None)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,camera_matrix=camera_matrix,dist_coeffs=dist_coeffs,image_size=np.array(image_size),rms=np.array(rms),pattern=np.array(pattern),square_size=np.array(args.square_size),accepted_images=np.array(accepted))
    print(f'[OK] Views used: {len(accepted)}/{len(paths)}')
    print(f'[OK] RMS reprojection error: {rms:.6f} (interpret with real, diverse images)')
    print(f'[OK] Saved calibration: {args.output.resolve()}')
    print('NOTE: Included generated checkerboard views are pipeline smoke-test fixtures, not valid evidence of real camera calibration.')
if __name__=='__main__': main()
