"""Track a user-selected region using an available OpenCV tracker."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
HERE=Path(__file__).resolve().parent
DEFAULT_SOURCE=str(HERE/'test_images'/'sample_motion.avi')
def parse_source(s): return int(s) if s.isdigit() else s

def create_tracker(name):
    candidates={'CSRT':['TrackerCSRT_create'],'KCF':['TrackerKCF_create'],'MIL':['TrackerMIL_create']}
    name=name.upper()
    if name not in candidates: raise ValueError('Tracker must be CSRT, KCF, or MIL.')
    for factory in candidates[name]:
        fn=getattr(cv2,factory,None)
        if fn: return fn()
        fn=getattr(getattr(cv2,'legacy',None),factory,None)
        if fn: return fn()
    raise RuntimeError(f'{name} tracker is not available in this OpenCV build. Install opencv-contrib-python or use --tracker MIL if provided.')

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--source',default=DEFAULT_SOURCE); ap.add_argument('--tracker',choices=['CSRT','KCF','MIL'],default='CSRT'); ap.add_argument('--roi',nargs=4,type=int,metavar=('X','Y','W','H'),help='Skip ROI picker and use explicit ROI'); ap.add_argument('--no-display',action='store_true'); args=ap.parse_args()
    cap=cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened(): raise SystemExit(f'ERROR: Cannot open source: {args.source}')
    ok,frame=cap.read()
    if not ok: cap.release(); raise SystemExit('ERROR: No readable frame.')
    if args.roi: roi=tuple(args.roi)
    elif not args.no_display:
        roi=tuple(map(int,cv2.selectROI('Select object and press ENTER',frame,False,False))); cv2.destroyWindow('Select object and press ENTER')
    else:
        cap.release(); raise SystemExit('In --no-display mode, pass --roi X Y W H.')
    if len(roi)!=4 or roi[2]<=0 or roi[3]<=0: cap.release(); raise SystemExit('ERROR: Invalid/empty ROI selection.')
    try: tracker=create_tracker(args.tracker)
    except Exception as exc: cap.release(); raise SystemExit(f'ERROR: {exc}')
    ok_init=tracker.init(frame,roi)
    if ok_init is False: cap.release(); raise SystemExit('ERROR: Tracker initialization failed.')
    frames=1
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            success,box=tracker.update(frame)
            label='TRACKING' if success else 'TARGET LOST'
            if success:
                x,y,w,h=map(int,box); cv2.rectangle(frame,(x,y),(x+w,y+h),(0,220,255),2)
            cv2.putText(frame,f'{args.tracker}: {label} | Q: quit',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.58,(0,255,255),2,cv2.LINE_AA)
            if not args.no_display and (cv2.imshow('Object Tracker',frame) is None):
                if cv2.waitKey(1)&0xFF==ord('q'): break
            frames+=1
    finally: cap.release(); cv2.destroyAllWindows()
    print(f'[OK] Tracker ran for {frames} frames.')
if __name__=='__main__': main()
