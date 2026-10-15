"""Contour-based object measurement. Physical units require known pixels-per-unit calibration."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
HERE=Path(__file__).resolve().parent
DEFAULT_IMAGE=HERE/'test_images'/'object_scene.png'
class ObjectMeasurement:
    def __init__(self,min_area=800,pixels_per_unit=None):
        self.min_area=float(min_area); self.pixels_per_unit=float(pixels_per_unit) if pixels_per_unit else None
        if self.pixels_per_unit is not None and self.pixels_per_unit<=0: raise ValueError('pixels_per_unit must be > 0')
    def process_frame(self,frame):
        if frame is None or frame.size==0: raise ValueError('Empty frame.')
        out=frame.copy(); gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gray=cv2.GaussianBlur(gray,(5,5),0)
        _,mask=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        detections=[]
        for c in sorted(contours,key=cv2.contourArea,reverse=True):
            area=float(cv2.contourArea(c))
            if area<self.min_area: continue
            x,y,w,h=cv2.boundingRect(c); detections.append({'bbox':(x,y,w,h),'area_px2':area,'width_px':w,'height_px':h})
            cv2.rectangle(out,(x,y),(x+w,y+h),(0,220,255),2)
            if self.pixels_per_unit:
                width=w/self.pixels_per_unit; height=h/self.pixels_per_unit; area_unit=area/(self.pixels_per_unit**2)
                label=f'{width:.2f} x {height:.2f} units | A={area_unit:.2f}'
            else: label=f'{w}px x {h}px | A={area:.0f}px2'
            cv2.putText(out,label,(x,max(20,y-8)),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),3,cv2.LINE_AA)
            cv2.putText(out,label,(x,max(20,y-8)),cv2.FONT_HERSHEY_SIMPLEX,.45,(25,25,25),1,cv2.LINE_AA)
        cv2.putText(out,f'OBJECTS: {len(detections)} | Q: quit',(14,28),cv2.FONT_HERSHEY_SIMPLEX,.62,(0,255,255),2,cv2.LINE_AA)
        return out,detections,mask

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--image',type=Path,default=DEFAULT_IMAGE); ap.add_argument('--source',help='Webcam index or video path'); ap.add_argument('--min-area',type=float,default=800); ap.add_argument('--pixels-per-unit',type=float,help='Calibration: pixels per real unit; omit to report pixels only'); ap.add_argument('--output',type=Path,default=HERE/'outputs'/'object_measurement_output.png'); ap.add_argument('--no-display',action='store_true'); args=ap.parse_args()
    detector=ObjectMeasurement(args.min_area,args.pixels_per_unit)
    if args.source is None:
        frame=cv2.imread(str(args.image))
        if frame is None: raise SystemExit(f'ERROR: Cannot read image: {args.image}')
        out,items,mask=detector.process_frame(frame); args.output.parent.mkdir(parents=True,exist_ok=True); cv2.imwrite(str(args.output),out); cv2.imwrite(str(args.output.with_name(args.output.stem+'_mask.png')),mask)
        print(f'[OK] Detected {len(items)} objects. Output: {args.output.resolve()}')
        if not args.no_display: cv2.imshow('Object Measurement',out); cv2.waitKey(0); cv2.destroyAllWindows()
        return
    source=int(args.source) if args.source.isdigit() else args.source; cap=cv2.VideoCapture(source)
    if not cap.isOpened(): raise SystemExit(f'ERROR: Cannot open source: {args.source}')
    count=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            if isinstance(source,int): frame=cv2.flip(frame,1)
            out,_,_=detector.process_frame(frame)
            if not args.no_display:
                cv2.imshow('Object Measurement',out)
                if cv2.waitKey(1)&0xFF==ord('q'): break
            count+=1
            if args.no_display and count>=300: break
    finally: cap.release(); cv2.destroyAllWindows()
    print(f'[OK] Processed {count} frames.')
if __name__=='__main__': main()
