"""Experimental eye-closure monitor using MediaPipe Face Mesh EAR.

This is a learning/demo tool, not a medical device or certified driver-safety system.
"""
from __future__ import annotations
import argparse
import time
import queue
import threading
from pathlib import Path
import cv2
import numpy as np
try:
    import mediapipe as mp
except ImportError as exc:
    raise SystemExit('MediaPipe is required. Install this folder\'s requirements.txt.') from exc
HERE=Path(__file__).resolve().parent
DEFAULT_IMAGE=HERE/'test_images'/'face_reference.png'
# MediaPipe Face Mesh six-point eye contours, ordered around each eye.
LEFT_EYE=(33,160,158,133,153,144)
RIGHT_EYE=(362,385,387,263,373,380)

def eye_aspect_ratio(points):
    p=np.asarray(points,dtype=np.float32)
    if p.shape!=(6,2): raise ValueError('EAR requires exactly six 2D points.')
    horizontal=float(np.linalg.norm(p[0]-p[3]))
    if horizontal<1e-6: return 0.0
    vertical1=float(np.linalg.norm(p[1]-p[5])); vertical2=float(np.linalg.norm(p[2]-p[4]))
    return (vertical1+vertical2)/(2.0*horizontal)

class DrowsinessMonitor:
    def __init__(self,threshold=0.21,closed_seconds=1.2,cooldown=3.0):
        self.threshold=threshold; self.closed_seconds=closed_seconds; self.cooldown=cooldown
        self.closed_since=None; self.last_alert=0.0; self.face_mesh=mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,max_num_faces=1,refine_landmarks=True,
            min_detection_confidence=.55,min_tracking_confidence=.55)
    def process(self,frame,now=None):
        now=time.monotonic() if now is None else now
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); result=self.face_mesh.process(rgb); out=frame.copy()
        status='NO FACE'; ear=None; alert=False
        if result.multi_face_landmarks:
            face=result.multi_face_landmarks[0].landmark; h,w=frame.shape[:2]
            def get_eye(indices): return np.asarray([(face[i].x*w,face[i].y*h) for i in indices],np.float32)
            lp=get_eye(LEFT_EYE); rp=get_eye(RIGHT_EYE)
            ear=(eye_aspect_ratio(lp)+eye_aspect_ratio(rp))/2.0
            for pts in [lp,rp]: cv2.polylines(out,[np.rint(pts).astype(np.int32)],True,(0,255,120),2,cv2.LINE_AA)
            if ear<self.threshold:
                self.closed_since=now if self.closed_since is None else self.closed_since
                closed_for=now-self.closed_since
                status='EYES CLOSED' if closed_for<self.closed_seconds else 'POSSIBLE DROWSINESS'
                if closed_for>=self.closed_seconds and now-self.last_alert>=self.cooldown:
                    alert=True; self.last_alert=now
            else:
                self.closed_since=None; status='EYES OPEN'
            cv2.putText(out,f'EAR: {ear:.3f}',(18,35),cv2.FONT_HERSHEY_SIMPLEX,.75,(0,255,255),2,cv2.LINE_AA)
        else:
            self.closed_since=None
        color=(0,0,255) if status=='POSSIBLE DROWSINESS' else (0,230,180)
        cv2.putText(out,status,(18,72),cv2.FONT_HERSHEY_SIMPLEX,.75,color,2,cv2.LINE_AA)
        cv2.putText(out,'DEMO ONLY - DO NOT RELY ON FOR DRIVING SAFETY',(18,frame.shape[0]-18),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
        return out,status,ear,alert
    def close(self): self.face_mesh.close()

class VoiceAlertWorker:
    """Optional non-blocking TTS worker; speech never stalls the vision loop."""
    def __init__(self):
        self.items=queue.Queue(maxsize=2)
        self.thread=threading.Thread(target=self._run,name='drowsiness-voice-alert',daemon=True)
        self.thread.start()
    def _run(self):
        try:
            import pyttsx3
            engine=pyttsx3.init()
            engine.setProperty('rate',155)
        except Exception as exc:
            print(f'[WARN] Voice alert unavailable: {exc}')
            return
        while True:
            message=self.items.get()
            if message is None: break
            try:
                engine.say(message); engine.runAndWait()
            except Exception as exc: print(f'[WARN] TTS playback failed: {exc}')
        try: engine.stop()
        except Exception: pass
    def alert(self):
        if self.items.empty():
            try: self.items.put_nowait('Possible drowsiness detected. Please take a safe break.')
            except queue.Full: pass
    def close(self):
        try: self.items.put_nowait(None)
        except queue.Full: pass
        self.thread.join(timeout=.5)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',default='0',help='Webcam index or video path')
    ap.add_argument('--image',type=Path,help='Optional still image test; cannot assess temporal drowsiness')
    ap.add_argument('--threshold',type=float,default=.21)
    ap.add_argument('--closed-seconds',type=float,default=1.2)
    ap.add_argument('--cooldown',type=float,default=3.0)
    ap.add_argument('--voice-alert',action='store_true',help='Speak an alert in a background worker when prolonged eye closure is detected')
    ap.add_argument('--no-display',action='store_true')
    args=ap.parse_args()
    monitor=DrowsinessMonitor(args.threshold,args.closed_seconds,args.cooldown)
    voice=VoiceAlertWorker() if args.voice_alert else None
    try:
        if args.image:
            frame=cv2.imread(str(args.image))
            if frame is None: raise SystemExit(f'ERROR: Could not read image: {args.image}')
            out,status,ear,_=monitor.process(frame)
            print(f'[INFO] Single frame only: status={status}, EAR={ear}. Temporal drowsiness cannot be inferred from one image.')
            if not args.no_display: cv2.imshow('Drowsiness Monitor - Single Frame',out); cv2.waitKey(0); cv2.destroyAllWindows()
            return
        source=int(args.source) if args.source.isdigit() else args.source
        cap=cv2.VideoCapture(source)
        if not cap.isOpened(): raise SystemExit(f'ERROR: Cannot open source: {args.source}')
        print('[INFO] Press Q to quit. Demo only; not a certified driver-safety system.')
        count=0
        try:
            while True:
                ok,frame=cap.read()
                if not ok: break
                if isinstance(source,int): frame=cv2.flip(frame,1)
                out,status,ear,alert=monitor.process(frame)
                if alert:
                    print('[ALERT] Possible prolonged eye closure detected; take a safe break if appropriate.')
                    if voice: voice.alert()
                if not args.no_display:
                    cv2.imshow('Drowsiness Monitor',out)
                    if cv2.waitKey(1)&0xFF==ord('q'): break
                count+=1
                if args.no_display and count>=300: break
        finally: cap.release(); cv2.destroyAllWindows()
        print(f'[OK] Processed {count} frames.')
    finally:
        monitor.close()
        if voice: voice.close()
if __name__=='__main__': main()
