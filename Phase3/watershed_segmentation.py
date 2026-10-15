"""Separate touching foreground objects using OpenCV watershed markers."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
HERE=Path(__file__).resolve().parent
DEFAULT_IMAGE=HERE/'test_images'/'watershed_scene.png'

def process(image):
    if image is None or image.size==0: raise ValueError('Input image is empty.')
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY); gray=cv2.GaussianBlur(gray,(5,5),0)
    _,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    opening=cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel,iterations=2)
    sure_bg=cv2.dilate(opening,kernel,iterations=3)
    dist=cv2.distanceTransform(opening,cv2.DIST_L2,5)
    _,sure_fg=cv2.threshold(dist,.42*dist.max(),255,0); sure_fg=np.uint8(sure_fg)
    unknown=cv2.subtract(sure_bg,sure_fg)
    count,markers=cv2.connectedComponents(sure_fg); markers=markers+1; markers[unknown==255]=0
    markers=cv2.watershed(image.copy(),markers)
    out=image.copy(); boundaries=markers==-1; out[boundaries]=(0,0,255)
    region_ids=[i for i in np.unique(markers) if i>1]
    cv2.putText(out,f'WATERSHED REGIONS: {len(region_ids)}',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2,cv2.LINE_AA)
    return out,markers,sure_fg

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--image',type=Path,default=DEFAULT_IMAGE); ap.add_argument('--output',type=Path,default=HERE/'outputs'/'watershed_output.png'); ap.add_argument('--show',action='store_true'); args=ap.parse_args()
    image=cv2.imread(str(args.image))
    if image is None: raise SystemExit(f'ERROR: Could not read: {args.image}')
    out,markers,mask=process(image); args.output.parent.mkdir(parents=True,exist_ok=True); cv2.imwrite(str(args.output),out); cv2.imwrite(str(args.output.with_name(args.output.stem+'_foreground.png')),mask)
    print(f'[OK] Saved watershed result: {args.output.resolve()}')
    if args.show: cv2.imshow('Watershed',out); cv2.imshow('Sure Foreground',mask); cv2.waitKey(0); cv2.destroyAllWindows()
if __name__=='__main__': main()
