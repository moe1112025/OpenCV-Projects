"""Decode QR codes from an image or webcam. Browser opening is opt-in and confirmed."""
from __future__ import annotations
import argparse
from pathlib import Path
import webbrowser
from urllib.parse import urlparse
import cv2
HERE=Path(__file__).resolve().parent
DEFAULT_IMAGE=HERE/'test_images'/'sample_qr.png'
class QRCodeScannerSystem:
    def __init__(self): self.detector=cv2.QRCodeDetector(); self.scanned_data=None
    def process(self,frame):
        if frame is None or frame.size==0: raise ValueError('Empty frame.')
        data,bbox,_=self.detector.detectAndDecode(frame); out=frame.copy()
        if data and bbox is not None:
            points=bbox.astype(int).reshape(-1,2)
            for i in range(len(points)): cv2.line(out,tuple(points[i]),tuple(points[(i+1)%len(points)]),(0,220,80),3,cv2.LINE_AA)
            cv2.putText(out,'QR DECODED',(15,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,190,0),2,cv2.LINE_AA)
        else:
            cv2.putText(out,'SHOW A QR CODE | Q: quit',(15,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,220,255),2,cv2.LINE_AA)
        return out,data or None

def safe_http_url(value):
    try:
        parsed=urlparse(value.strip())
        return parsed.scheme.lower() in {'http','https'} and bool(parsed.netloc) and not any(ch.isspace() for ch in value)
    except Exception: return False

def maybe_open_url(data,enabled):
    if not enabled: return
    if not safe_http_url(data):
        print('[BLOCKED] Payload is not a plain HTTP(S) URL. It was not opened.')
        return
    try: answer=input(f'Open this decoded URL in your default browser?\n{data}\nType Y to confirm: ').strip().lower()
    except EOFError: answer=''
    if answer=='y': webbrowser.open(data,new=2); print('[OK] Browser launch requested.')
    else: print('[INFO] URL not opened.')

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--image',type=Path,default=DEFAULT_IMAGE); ap.add_argument('--source',help='Webcam index (0) or video path'); ap.add_argument('--open-url',action='store_true',help='Ask before opening a decoded HTTP(S) URL'); ap.add_argument('--output',type=Path,default=HERE/'outputs'/'qr_scan_output.png'); ap.add_argument('--no-display',action='store_true'); args=ap.parse_args(); scanner=QRCodeScannerSystem()
    if args.source is None:
        frame=cv2.imread(str(args.image))
        if frame is None: raise SystemExit(f'ERROR: Cannot read image: {args.image}')
        out,data=scanner.process(frame); args.output.parent.mkdir(parents=True,exist_ok=True); cv2.imwrite(str(args.output),out)
        if data: print(f'[DECODED] {data}')
        else: print('[INFO] No QR code decoded from this image.')
        print(f'[OK] Output: {args.output.resolve()}')
        if args.open_url and data: maybe_open_url(data,True)
        if not args.no_display: cv2.imshow('QR Code Scanner',out); cv2.waitKey(0); cv2.destroyAllWindows()
        return
    source=int(args.source) if args.source.isdigit() else args.source; cap=cv2.VideoCapture(source)
    if not cap.isOpened(): raise SystemExit(f'ERROR: Cannot open source: {args.source}')
    data=None
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            if isinstance(source,int): frame=cv2.flip(frame,1)
            out,decoded=scanner.process(frame)
            if decoded: data=decoded; print(f'[DECODED] {data}')
            if not args.no_display:
                cv2.imshow('QR Code Scanner',out)
                if cv2.waitKey(1)&0xFF==ord('q'): break
            if data: break
    finally: cap.release(); cv2.destroyAllWindows()
    if data and args.open_url: maybe_open_url(data,True)
    if not data: print('[INFO] No QR code was decoded.')
if __name__=='__main__': main()
