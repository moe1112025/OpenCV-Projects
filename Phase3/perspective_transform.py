"""Detect the largest quadrilateral and rectify it with a perspective transform."""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
HERE=Path(__file__).resolve().parent
DEFAULT_IMAGE=HERE/'test_images'/'document_scene.png'

def order_points(points):
    p=np.asarray(points,dtype=np.float32).reshape(4,2); sums=p.sum(axis=1); diffs=np.diff(p,axis=1).reshape(-1)
    return np.array([p[np.argmin(sums)],p[np.argmin(diffs)],p[np.argmax(sums)],p[np.argmax(diffs)]],np.float32)

def detect_quad(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY); blur=cv2.GaussianBlur(gray,(5,5),0)
    edges=cv2.Canny(blur,50,160); edges=cv2.dilate(edges,None,iterations=1)
    contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    candidates=sorted(contours,key=cv2.contourArea,reverse=True)
    for c in candidates[:30]:
        if cv2.contourArea(c)<0.08*image.shape[0]*image.shape[1]: continue
        peri=cv2.arcLength(c,True); approx=cv2.approxPolyDP(c,.02*peri,True)
        if len(approx)==4 and cv2.isContourConvex(approx): return order_points(approx.reshape(4,2))
    return None

def transform(image,points=None):
    if points is None: points=detect_quad(image)
    if points is None: raise ValueError('No quadrilateral found. Use --points x1 y1 x2 y2 x3 y3 x4 y4 (TL TR BR BL).')
    rect=order_points(points); tl,tr,br,bl=rect
    width=max(int(np.linalg.norm(br-bl)),int(np.linalg.norm(tr-tl))); height=max(int(np.linalg.norm(tr-br)),int(np.linalg.norm(tl-bl)))
    if width<2 or height<2: raise ValueError('Detected quadrilateral has invalid dimensions.')
    dest=np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],np.float32)
    matrix=cv2.getPerspectiveTransform(rect,dest); return cv2.warpPerspective(image,matrix,(width,height)),rect

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--image',type=Path,default=DEFAULT_IMAGE); ap.add_argument('--points',nargs=8,type=float,help='Manual points in TL TR BR BL order: x1 y1 x2 y2 x3 y3 x4 y4'); ap.add_argument('--output',type=Path,default=HERE/'outputs'/'perspective_corrected.png'); ap.add_argument('--show',action='store_true'); args=ap.parse_args()
    image=cv2.imread(str(args.image))
    if image is None: raise SystemExit(f'ERROR: Could not read image: {args.image}')
    pts=np.asarray(args.points,dtype=np.float32).reshape(4,2) if args.points else None
    try: result,rect=transform(image,pts)
    except ValueError as exc: raise SystemExit(f'ERROR: {exc}')
    preview=image.copy(); cv2.polylines(preview,[np.rint(rect).astype(np.int32)],True,(0,255,0),3)
    args.output.parent.mkdir(parents=True,exist_ok=True); cv2.imwrite(str(args.output),result); cv2.imwrite(str(args.output.with_name(args.output.stem+'_detected_quad.png')),preview)
    print(f'[OK] Rectified dimensions: {result.shape[1]} x {result.shape[0]} px'); print(f'[OK] Output: {args.output.resolve()}')
    if args.show: cv2.imshow('Detected Quadrilateral',preview); cv2.imshow('Perspective Corrected',result); cv2.waitKey(0); cv2.destroyAllWindows()
if __name__=='__main__': main()
