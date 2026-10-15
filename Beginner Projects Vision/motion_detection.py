"""Detect motion in a video stream using MOG2 background subtraction."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
HERE=Path(__file__).resolve().parent
DEFAULT_SOURCE=str(HERE/'test_images'/'sample_motion.avi')
def parse_source(s): return int(s) if s.isdigit() else s

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--source',default=DEFAULT_SOURCE); ap.add_argument('--min-area',type=float,default=500); ap.add_argument('--no-display',action='store_true'); ap.add_argument('--max-frames',type=int,default=400); args=ap.parse_args()
    cap=cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened(): raise SystemExit(f'ERROR: Cannot open source: {args.source}')
    bg=cv2.createBackgroundSubtractorMOG2(history=180,varThreshold=28,detectShadows=True); count=0
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            mask=bg.apply(frame); mask=cv2.threshold(mask,220,255,cv2.THRESH_BINARY)[1]
            mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel); mask=cv2.dilate(mask,kernel,iterations=2)
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); objects=0
            for c in contours:
                if cv2.contourArea(c)<args.min_area: continue
                x,y,w,h=cv2.boundingRect(c); cv2.rectangle(frame,(x,y),(x+w,y+h),(0,220,255),2); objects+=1
            cv2.putText(frame,f'MOTION REGIONS: {objects} | Q: quit',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.58,(0,255,255),2,cv2.LINE_AA)
            if not args.no_display:
                cv2.imshow('Motion Detection',frame); cv2.imshow('Foreground Mask',mask)
                if cv2.waitKey(1)&0xFF==ord('q'): break
            count+=1
            if args.no_display and count>=args.max_frames: break
    finally: cap.release(); cv2.destroyAllWindows()
    print(f'[OK] Motion detection processed {count} frames.')
if __name__=='__main__': main()
