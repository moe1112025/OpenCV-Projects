"""Haar-cascade face and eye detector with image and webcam modes."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
HERE=Path(__file__).resolve().parent
DEFAULT_IMAGE=HERE/'test_images'/'face_reference.png'
def resolve_cascade(filename,opencv_name):
    candidates=[HERE/filename,Path(cv2.data.haarcascades)/opencv_name]
    for path in candidates:
        if path.is_file(): return str(path)
    raise FileNotFoundError(f'Cascade not found: {filename}. Keep the XML beside this script or reinstall opencv-python.')
class FaceEyeDetector:
    def __init__(self,face_path=None,eye_path=None):
        face_path=face_path or resolve_cascade('haarcascade_frontalface_default.xml','haarcascade_frontalface_default.xml')
        eye_path=eye_path or resolve_cascade('haarcascade_eye.xml','haarcascade_eye.xml')
        self.face=cv2.CascadeClassifier(str(face_path)); self.eye=cv2.CascadeClassifier(str(eye_path))
        if self.face.empty(): raise ValueError(f'Invalid face cascade: {face_path}')
        if self.eye.empty(): raise ValueError(f'Invalid eye cascade: {eye_path}')
    def process(self,frame):
        if frame is None or frame.size==0: raise ValueError('Empty frame.')
        out=frame.copy(); gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gray=cv2.equalizeHist(gray)
        faces=self.face.detectMultiScale(gray,scaleFactor=1.12,minNeighbors=5,minSize=(45,45))
        eye_count=0
        for x,y,w,h in faces:
            cv2.rectangle(out,(x,y),(x+w,y+h),(255,90,30),2)
            cv2.putText(out,'FACE',(x,max(20,y-8)),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,90,30),2,cv2.LINE_AA)
            roi=gray[y:y+h,x:x+w]; roi_out=out[y:y+h,x:x+w]
            eyes=self.eye.detectMultiScale(roi,scaleFactor=1.12,minNeighbors=6,minSize=(12,12))
            for ex,ey,ew,eh in eyes:
                cv2.rectangle(roi_out,(ex,ey),(ex+ew,ey+eh),(0,210,255),2); eye_count+=1
        cv2.putText(out,f'FACES: {len(faces)} | EYE REGIONS: {eye_count}',(15,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,255),2,cv2.LINE_AA)
        return out,len(faces),eye_count

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--image',type=Path,default=DEFAULT_IMAGE); ap.add_argument('--source',help='Webcam index such as 0 or a video path'); ap.add_argument('--output',type=Path,default=HERE/'outputs'/'face_eye_output.png'); ap.add_argument('--no-display',action='store_true'); args=ap.parse_args()
    detector=FaceEyeDetector()
    if args.source is None:
        frame=cv2.imread(str(args.image))
        if frame is None: raise SystemExit(f'ERROR: Cannot read image: {args.image}')
        out,nf,ne=detector.process(frame); args.output.parent.mkdir(parents=True,exist_ok=True); cv2.imwrite(str(args.output),out)
        print(f'[OK] faces={nf}, eye_regions={ne} | saved: {args.output.resolve()}')
        if not args.no_display: cv2.imshow('Face and Eye Detection',out); cv2.waitKey(0); cv2.destroyAllWindows()
        return
    source=int(args.source) if args.source.isdigit() else args.source; cap=cv2.VideoCapture(source)
    if not cap.isOpened(): raise SystemExit(f'ERROR: Cannot open source: {args.source}')
    count=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            if isinstance(source,int): frame=cv2.flip(frame,1)
            out,_,_=detector.process(frame)
            if not args.no_display:
                cv2.imshow('Face and Eye Detection',out)
                if cv2.waitKey(1)&0xFF==ord('q'): break
            count+=1
            if args.no_display and count>=300: break
    finally: cap.release(); cv2.destroyAllWindows()
    print(f'[OK] Processed {count} frames.')
if __name__=='__main__': main()
