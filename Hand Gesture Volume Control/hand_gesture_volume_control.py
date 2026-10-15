"""Control Windows media volume with a thumb-index pinch gesture.

One media-key press occurs only when the gesture state changes into UP/DOWN;
there is no per-frame key spam. This is a local webcam/keyboard automation demo.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
try:
    import mediapipe as mp
except ImportError as exc:
    raise SystemExit('MediaPipe is required. Install this folder\'s requirements.txt.') from exc
try:
    import pyautogui
except Exception as exc:
    raise SystemExit(f'PyAutoGUI could not initialize: {exc}. Check its platform/display requirements.') from exc

HERE=Path(__file__).resolve().parent

def distance(a,b): return float(np.linalg.norm(np.asarray(a,dtype=np.float32)-np.asarray(b,dtype=np.float32)))

def classify_gesture(landmarks,width,height,up_ratio=.58,down_ratio=.28):
    pts={i:(landmarks[i].x*width,landmarks[i].y*height) for i in (4,5,8,17)}
    pinch=distance(pts[4],pts[8]); palm=max(1.0,distance(pts[5],pts[17])); ratio=pinch/palm
    if ratio>=up_ratio: state='UP'
    elif ratio<=down_ratio: state='DOWN'
    else: state='NEUTRAL'
    return state,ratio,pts

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--source',default='0',help='Camera index or video file path'); ap.add_argument('--disable-controls',action='store_true',help='Track and print gestures but do not send media keys'); ap.add_argument('--min-command-interval',type=float,default=.35); ap.add_argument('--no-display',action='store_true'); args=ap.parse_args()
    if args.min_command_interval<0: raise SystemExit('--min-command-interval cannot be negative.')
    source=int(args.source) if args.source.isdigit() else args.source
    cap=cv2.VideoCapture(source)
    if not cap.isOpened(): raise SystemExit(f'ERROR: Could not open camera/video: {args.source}')
    mp_hands=mp.solutions.hands; drawing=mp.solutions.drawing_utils
    last_state='NEUTRAL'; last_command_time=0.0; frame_count=0
    pyautogui.FAILSAFE=True
    print('[INFO] UP gesture = volume up, DOWN pinch = volume down, neutral = no action. Press Q or ESC to exit.')
    try:
        with mp_hands.Hands(static_image_mode=False,max_num_hands=1,min_detection_confidence=.65,min_tracking_confidence=.6) as hands:
            while True:
                ok,frame=cap.read()
                if not ok: break
                if isinstance(source,int): frame=cv2.flip(frame,1)
                h,w=frame.shape[:2]; rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); result=hands.process(rgb)
                state='NO HAND'; ratio=None
                if result.multi_hand_landmarks:
                    hand=result.multi_hand_landmarks[0]; drawing.draw_landmarks(frame,hand,mp_hands.HAND_CONNECTIONS)
                    state,ratio,pts=classify_gesture(hand.landmark,w,h)
                    # Mark tips used by the gesture state machine.
                    for idx in (4,8):
                        lm=hand.landmark[idx]; cv2.circle(frame,(int(lm.x*w),int(lm.y*h)),8,(0,255,255),2,cv2.LINE_AA)
                    now=time.monotonic()
                    if state in {'UP','DOWN'} and state!=last_state and now-last_command_time>=args.min_command_interval:
                        if args.disable_controls:
                            print(f'[DRY RUN] Gesture state: {state}; media key not sent.')
                        else:
                            try:
                                pyautogui.press('volumeup' if state=='UP' else 'volumedown')
                                print(f'[ACTION] {state}: sent one media-key press.')
                            except Exception as exc:
                                print(f'[WARN] Could not send media key: {exc}')
                        last_command_time=now
                    last_state=state
                else:
                    last_state='NO HAND'
                cv2.putText(frame,f'GESTURE: {state}',(15,30),cv2.FONT_HERSHEY_SIMPLEX,.75,(0,255,255),2,cv2.LINE_AA)
                if ratio is not None: cv2.putText(frame,f'PINCH / PALM: {ratio:.2f}',(15,61),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2,cv2.LINE_AA)
                cv2.putText(frame,'Q / ESC: quit',(15,h-18),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
                if not args.no_display:
                    cv2.imshow('Hand Gesture Volume Control',frame)
                    key=cv2.waitKey(1)&0xFF
                    if key in (ord('q'),27): break
                frame_count+=1
                if args.no_display and frame_count>=300: break
    finally: cap.release(); cv2.destroyAllWindows()
    print(f'[OK] Processed {frame_count} frames.')
if __name__=='__main__': main()
