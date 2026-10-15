"""Track moving bright objects and draw a centroid trajectory."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
HERE=Path(__file__).resolve().parent
DEFAULT_SOURCE=str(HERE / "test_images" / "sample_motion.avi")

def parse_source(value):
    return int(value) if value.isdigit() else value

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source",default=DEFAULT_SOURCE,help="Video path, or camera index such as 0")
    ap.add_argument("--min-area",type=float,default=250)
    ap.add_argument("--max-points",type=int,default=80)
    ap.add_argument("--no-display",action="store_true")
    args=ap.parse_args(); source=parse_source(args.source)
    cap=cv2.VideoCapture(source)
    if not cap.isOpened(): raise SystemExit(f"ERROR: Cannot open video/camera: {args.source}")
    subtractor=cv2.createBackgroundSubtractorMOG2(history=150,varThreshold=24,detectShadows=False)
    trail=[]; frame_no=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            frame_no+=1
            fg=subtractor.apply(frame)
            fg=cv2.threshold(fg,220,255,cv2.THRESH_BINARY)[1]
            fg=cv2.morphologyEx(fg,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
            contours,_=cv2.findContours(fg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            candidates=[c for c in contours if cv2.contourArea(c)>=args.min_area]
            if candidates:
                c=max(candidates,key=cv2.contourArea); m=cv2.moments(c)
                if m["m00"]:
                    point=(int(m["m10"]/m["m00"]),int(m["m01"]/m["m00"]))
                    trail.append(point); trail=trail[-max(1,args.max_points):]
                    x,y,w,h=cv2.boundingRect(c); cv2.rectangle(frame,(x,y),(x+w,y+h),(0,220,255),2)
                    cv2.circle(frame,point,5,(0,0,255),-1)
            for i in range(1,len(trail)):
                cv2.line(frame,trail[i-1],trail[i],(255,220,0),2,cv2.LINE_AA)
            cv2.putText(frame,"CENTROID TRAJECTORY | Q: quit",(12,28),cv2.FONT_HERSHEY_SIMPLEX,.58,(0,255,255),2,cv2.LINE_AA)
            if not args.no_display:
                cv2.imshow("Centroid Trajectory",frame)
                if cv2.waitKey(1)&0xFF==ord('q'): break
            elif frame_no>=300: break
    finally:
        cap.release(); cv2.destroyAllWindows()
    if frame_no==0: print("[WARN] No readable frames were received.")
    else: print(f"[OK] Processed {frame_no} frames.")
if __name__=="__main__": main()
