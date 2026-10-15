"""Real-time multi-color detector with a still-image test mode."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
HERE=Path(__file__).resolve().parent
DEFAULT_IMAGE=HERE/'test_images'/'color_scene.png'

class ColorDetector:
    def __init__(self,min_area=500):
        self.min_area=float(min_area)
        self.kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        # OpenCV hue is 0..179; red wraps around zero and therefore has two ranges.
        self.ranges={
          'RED': [((0,90,65),(10,255,255)),((170,90,65),(179,255,255))],
          'GREEN': [((35,65,55),(90,255,255))],
          'BLUE': [((92,70,45),(132,255,255))],
        }
        self.colors={'RED':(0,0,255),'GREEN':(0,255,0),'BLUE':(255,80,0)}
    def process(self,frame):
        if frame is None or frame.size==0: raise ValueError('Empty frame received.')
        out=frame.copy(); hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV); summary={}
        for name,ranges in self.ranges.items():
            mask=np.zeros(hsv.shape[:2],np.uint8)
            for low,high in ranges:
                mask=cv2.bitwise_or(mask,cv2.inRange(hsv,np.array(low,np.uint8),np.array(high,np.uint8)))
            mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,self.kernel)
            mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,self.kernel,iterations=2)
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            hits=0
            for c in sorted(contours,key=cv2.contourArea,reverse=True):
                area=cv2.contourArea(c)
                if area<self.min_area: continue
                x,y,w,h=cv2.boundingRect(c); color=self.colors[name]
                cv2.rectangle(out,(x,y),(x+w,y+h),color,2)
                cv2.putText(out,f'{name} | {area:.0f}px2',(x,max(22,y-8)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2,cv2.LINE_AA)
                hits+=1
            summary[name]=hits
        cv2.putText(out,'MULTI-COLOR DETECTOR',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2,cv2.LINE_AA)
        return out,summary

def source_value(value): return int(value) if value.isdigit() else value

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--image',type=Path,default=DEFAULT_IMAGE,help='Still image input; use --source for webcam/video.')
    ap.add_argument('--source',help='Camera index (0) or video path; overrides --image')
    ap.add_argument('--min-area',type=float,default=500)
    ap.add_argument('--output',type=Path,default=HERE/'outputs'/'color_detector_output.png')
    ap.add_argument('--no-display',action='store_true')
    args=ap.parse_args(); detector=ColorDetector(args.min_area)
    if args.source is None:
        frame=cv2.imread(str(args.image))
        if frame is None: raise SystemExit(f'ERROR: Cannot read input image: {args.image}')
        out,summary=detector.process(frame); args.output.parent.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(args.output),out): raise SystemExit('ERROR: Could not write output image.')
        print('[OK] Detected contours: '+', '.join(f'{k}={v}' for k,v in summary.items()))
        print(f'[OK] Output saved: {args.output.resolve()}')
        if not args.no_display:
            cv2.imshow('Color Detector',out); cv2.waitKey(0); cv2.destroyAllWindows()
        return
    cap=cv2.VideoCapture(source_value(args.source))
    if not cap.isOpened(): raise SystemExit(f'ERROR: Cannot open camera/video source: {args.source}')
    count=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            frame=cv2.flip(frame,1) if isinstance(source_value(args.source),int) else frame
            out,summary=detector.process(frame)
            if not args.no_display:
                cv2.imshow('Color Detector',out)
                if cv2.waitKey(1)&0xFF==ord('q'): break
            count+=1
            if args.no_display and count>=300: break
    finally: cap.release(); cv2.destroyAllWindows()
    print(f'[OK] Processed {count} frames. Press Q in the display to exit.')
if __name__=='__main__': main()
