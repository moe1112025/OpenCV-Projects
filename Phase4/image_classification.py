"""Train a shape classifier using contour geometry and a Random Forest."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
HERE=Path(__file__).resolve().parent
DATA=HERE/'test_images'/'shapes_dataset'
MODEL=HERE/'outputs'/'geometry_rf_model.joblib'

def extract_features(image):
    if image is None or image.size==0: raise ValueError('Empty input image.')
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image.copy()
    blur=cv2.GaussianBlur(gray,(5,5),0)
    # Shapes are darker than the synthetic dataset background; Otsu selects a threshold.
    _,mask=cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    h,w=gray.shape[:2]; image_area=float(h*w)
    candidates=[c for c in contours if 100<cv2.contourArea(c)<.65*image_area]
    if not candidates: raise ValueError('No foreground contour found. Use a high-contrast object image.')
    c=max(candidates,key=cv2.contourArea); area=float(cv2.contourArea(c)); perimeter=float(cv2.arcLength(c,True))
    x,y,bw,bh=cv2.boundingRect(c); rect_area=max(1,bw*bh)
    hull=cv2.convexHull(c); hull_area=max(float(cv2.contourArea(hull)),1.0)
    moments=cv2.moments(c); hu=cv2.HuMoments(moments).flatten()
    hu=np.sign(hu)*np.log10(np.abs(hu)+1e-12)
    circularity=4*np.pi*area/(perimeter*perimeter) if perimeter else 0.0
    base=np.array([bw/max(1,bh),area/rect_area,area/hull_area,circularity,area/image_area],np.float32)
    return np.concatenate([base,hu.astype(np.float32)])

def load_dataset(root,split):
    X=[]; y=[]
    base=root/split
    if not base.is_dir(): raise FileNotFoundError(f'Dataset split not found: {base}')
    for class_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for file in sorted(class_dir.iterdir()):
            if file.suffix.lower() not in {'.png','.jpg','.jpeg','.bmp'}: continue
            im=cv2.imread(str(file))
            if im is None: print(f'[WARN] Skipping unreadable image: {file}'); continue
            try: X.append(extract_features(im)); y.append(class_dir.name)
            except ValueError as exc: print(f'[WARN] {file.name}: {exc}')
    if not X: raise ValueError(f'No valid samples under {base}')
    return np.vstack(X),np.asarray(y)

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--mode',choices=['train','predict'],default='train'); ap.add_argument('--dataset',type=Path,default=DATA); ap.add_argument('--image',type=Path); ap.add_argument('--model',type=Path,default=MODEL); ap.add_argument('--report',type=Path,default=HERE/'outputs'/'geometry_rf_report.json'); args=ap.parse_args()
    args.model.parent.mkdir(parents=True,exist_ok=True)
    if args.mode=='train':
        X_train,y_train=load_dataset(args.dataset,'train'); X_test,y_test=load_dataset(args.dataset,'test')
        model=RandomForestClassifier(n_estimators=250,max_features='sqrt',class_weight='balanced',random_state=42,n_jobs=1)
        model.fit(X_train,y_train); pred=model.predict(X_test); score=float(accuracy_score(y_test,pred))
        report={'model':'Contour geometry + RandomForest','train_samples':int(len(y_train)),'test_samples':int(len(y_test)),'classes':list(model.classes_),'accuracy':score,'classification_report':classification_report(y_test,pred,output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(y_test,pred,labels=model.classes_).tolist(),'note':'Synthetic geometric shape dataset; not a pretrained general-purpose object classifier.'}
        joblib.dump({'model':model,'feature_version':1,'classes':list(model.classes_)},args.model)
        args.report.parent.mkdir(parents=True,exist_ok=True); args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(f'[OK] Trained geometry RF on {len(y_train)} images; held-out accuracy={score:.3f}.')
        print(f'[OK] Model: {args.model.resolve()}'); print(f'[OK] Metrics: {args.report.resolve()}')
    else:
        if not args.image: raise SystemExit('ERROR: --image is required when --mode predict.')
        if not args.model.is_file(): raise SystemExit(f'ERROR: Model not found: {args.model}. Run --mode train first.')
        im=cv2.imread(str(args.image))
        if im is None: raise SystemExit(f'ERROR: Could not read image: {args.image}')
        try: bundle=joblib.load(args.model); feature=extract_features(im).reshape(1,-1); model=bundle['model']
        except Exception as exc: raise SystemExit(f'ERROR: Could not load/predict with model: {exc}') from exc
        pred=model.predict(feature)[0]
        conf=float(np.max(model.predict_proba(feature)[0])) if hasattr(model,'predict_proba') else None
        print(f'[PREDICTION] {pred}'+(f' | model probability={conf:.3f}' if conf is not None else ''))
if __name__=='__main__': main()
