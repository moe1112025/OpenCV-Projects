"""Segment a foreground region with GrabCut and save image/mask outputs."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
HERE=Path(__file__).resolve().parent
DEFAULT_IMAGE=HERE/'test_images'/'grabcut_scene.png'

def process(image,rect=None,iterations=5):
    if image is None or image.size==0: raise ValueError('Input image is empty.')
    h,w=image.shape[:2]
    if rect is None: rect=(int(.06*w),int(.06*h),int(.88*w),int(.88*h))
    x,y,rw,rh=map(int,rect)
    if rw<=1 or rh<=1 or x<0 or y<0 or x+rw>w or y+rh>h: raise ValueError('GrabCut rectangle must lie inside the image and have positive size.')
    mask=np.zeros((h,w),np.uint8); bgd=np.zeros((1,65),np.float64); fgd=np.zeros((1,65),np.float64)
    cv2.grabCut(image,mask,(x,y,rw,rh),bgd,fgd,iterations,cv2.GC_INIT_WITH_RECT)
    binary=np.where((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD),255,0).astype(np.uint8)
    segmented=cv2.bitwise_and(image,image,mask=binary)
    return segmented,binary

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--image',type=Path,default=DEFAULT_IMAGE); ap.add_argument('--rect',nargs=4,type=int,metavar=('X','Y','W','H')); ap.add_argument('--iterations',type=int,default=5); ap.add_argument('--output',type=Path,default=HERE/'outputs'/'grabcut_segmented.png'); ap.add_argument('--show',action='store_true'); args=ap.parse_args()
    if args.iterations<1: raise SystemExit('--iterations must be at least 1')
    image=cv2.imread(str(args.image))
    if image is None: raise SystemExit(f'ERROR: Could not read image: {args.image}')
    segmented,mask=process(image,args.rect,args.iterations); args.output.parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(args.output),segmented); mask_path=args.output.with_name(args.output.stem+'_mask.png'); cv2.imwrite(str(mask_path),mask)
    print(f'[OK] Segmented image: {args.output.resolve()}'); print(f'[OK] Mask: {mask_path.resolve()}')
    if args.show: cv2.imshow('GrabCut Segmentation',segmented); cv2.imshow('Foreground Mask',mask); cv2.waitKey(0); cv2.destroyAllWindows()
if __name__=='__main__': main()
