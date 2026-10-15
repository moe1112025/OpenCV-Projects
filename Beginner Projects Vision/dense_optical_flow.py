"""Visualize dense optical flow using Farneback's algorithm."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
HERE=Path(__file__).resolve().parent
DEFAULT_SOURCE=str(HERE/"test_images"/"sample_motion.avi")
def parse_source(s): return int(s) if s.isdigit() else s

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--source',default=DEFAULT_SOURCE); ap.add_argument('--no-display',action='store_true'); ap.add_argument('--max-frames',type=int,default=300); args=ap.parse_args()
    cap=cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened(): raise SystemExit(f"ERROR: Cannot open source: {args.source}")
    ok,prev=cap.read()
    if not ok: cap.release(); raise SystemExit('ERROR: Source contains no readable frames.')
    prev_gray=cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY); idx=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            flow=cv2.calcOpticalFlowFarneback(prev_gray,gray,None,0.5,3,15,3,5,1.2,0)
            mag,ang=cv2.cartToPolar(flow[...,0],flow[...,1])
            hsv=np.zeros_like(frame); hsv[...,1]=255
            hsv[...,0]=(ang*180/np.pi/2).astype(np.uint8)
            hsv[...,2]=cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
            visual=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            cv2.putText(visual,'DENSE OPTICAL FLOW | Q: quit',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),2,cv2.LINE_AA)
            if not args.no_display:
                cv2.imshow('Dense Optical Flow',visual)
                if cv2.waitKey(1)&0xFF==ord('q'): break
            idx+=1; prev_gray=gray
            if args.no_display and idx>=args.max_frames: break
    finally: cap.release(); cv2.destroyAllWindows()
    print(f'[OK] Dense flow processed {idx} frame pairs.')
if __name__=='__main__': main()
