"""Track sparse corner features with Lucas-Kanade optical flow."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
HERE=Path(__file__).resolve().parent
DEFAULT_SOURCE=str(HERE/'test_images'/'sample_motion.avi')
def parse_source(s): return int(s) if s.isdigit() else s

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--source',default=DEFAULT_SOURCE); ap.add_argument('--no-display',action='store_true'); ap.add_argument('--max-frames',type=int,default=300); args=ap.parse_args()
    cap=cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened(): raise SystemExit(f'ERROR: Cannot open source: {args.source}')
    ok,old_frame=cap.read()
    if not ok: cap.release(); raise SystemExit('ERROR: Source contains no frames.')
    old_gray=cv2.cvtColor(old_frame,cv2.COLOR_BGR2GRAY)
    p0=cv2.goodFeaturesToTrack(old_gray,mask=None,maxCorners=250,qualityLevel=.25,minDistance=7,blockSize=7)
    trail_mask=np.zeros_like(old_frame); frame_count=0
    lk=dict(winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            if p0 is not None and len(p0):
                p1,st,err=cv2.calcOpticalFlowPyrLK(old_gray,gray,p0,None,**lk)
                if p1 is not None and st is not None:
                    good_new=p1[st.reshape(-1)==1]; good_old=p0[st.reshape(-1)==1]
                    for a,b in zip(good_new,good_old):
                        x1,y1=np.rint(a).astype(int).reshape(-1); x0,y0=np.rint(b).astype(int).reshape(-1)
                        cv2.line(trail_mask,(x1,y1),(x0,y0),(0,220,255),2)
                        cv2.circle(frame,(x1,y1),3,(0,0,255),-1)
                    p0=good_new.reshape(-1,1,2) if len(good_new) else None
            if p0 is None or len(p0)<8:
                p0=cv2.goodFeaturesToTrack(gray,mask=None,maxCorners=250,qualityLevel=.25,minDistance=7,blockSize=7)
            visual=cv2.add(frame,trail_mask)
            cv2.putText(visual,'SPARSE OPTICAL FLOW | Q: quit',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.58,(0,255,255),2,cv2.LINE_AA)
            if not args.no_display:
                cv2.imshow('Sparse Optical Flow',visual)
                if cv2.waitKey(1)&0xFF==ord('q'): break
            old_gray=gray; frame_count+=1
            if args.no_display and frame_count>=args.max_frames: break
    finally: cap.release(); cv2.destroyAllWindows()
    print(f'[OK] Sparse flow processed {frame_count} frame pairs.')
if __name__=='__main__': main()
