"""Train/evaluate an HOG + SVM image classifier on the bundled shape dataset."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
import joblib
HERE=Path(__file__).resolve().parent
DATA=HERE/'test_images'/'shapes_dataset'
MODEL=HERE/'outputs'/'hog_svm_model.joblib'

class HOGFeatureExtractor:
    def __init__(self,size=(64,64)):
        self.size=size
        self.hog=cv2.HOGDescriptor(size,(16,16),(8,8),(8,8),9)
    def extract(self,image):
        if image is None or image.size==0: raise ValueError('Empty input image.')
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
        gray=cv2.resize(gray,self.size,interpolation=cv2.INTER_AREA)
        return self.hog.compute(gray).reshape(-1).astype(np.float32)

def load_dataset(root,split,extractor):
    features=[]; labels=[]
    split_dir=root/split
    if not split_dir.is_dir(): raise FileNotFoundError(f'Dataset split not found: {split_dir}')
    class_dirs=sorted(p for p in split_dir.iterdir() if p.is_dir())
    if len(class_dirs)<2: raise ValueError(f'Expected at least two class folders under {split_dir}')
    for cls in class_dirs:
        for path in sorted(cls.iterdir()):
            if path.suffix.lower() not in {'.png','.jpg','.jpeg','.bmp'}: continue
            image=cv2.imread(str(path))
            if image is None: print(f'[WARN] Skipping unreadable image: {path}'); continue
            features.append(extractor.extract(image)); labels.append(cls.name)
    if not features: raise ValueError(f'No readable images under {split_dir}')
    return np.vstack(features),np.asarray(labels)

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--mode',choices=['train','predict'],default='train'); ap.add_argument('--dataset',type=Path,default=DATA); ap.add_argument('--image',type=Path); ap.add_argument('--model',type=Path,default=MODEL); ap.add_argument('--report',type=Path,default=HERE/'outputs'/'hog_svm_report.json'); args=ap.parse_args()
    extractor=HOGFeatureExtractor(); args.model.parent.mkdir(parents=True,exist_ok=True)
    if args.mode=='train':
        X_train,y_train=load_dataset(args.dataset,'train',extractor); X_test,y_test=load_dataset(args.dataset,'test',extractor)
        if set(y_train)!=set(y_test): raise SystemExit('ERROR: Train and test splits must contain the same class folders.')
        model=SVC(kernel='rbf',C=5.0,gamma='scale',class_weight='balanced',probability=True,random_state=42)
        model.fit(X_train,y_train); predicted=model.predict(X_test); accuracy=float(accuracy_score(y_test,predicted))
        report={'model':'HOG + RBF SVM','train_samples':int(len(y_train)),'test_samples':int(len(y_test)),'classes':list(model.classes_),'accuracy':accuracy,'classification_report':classification_report(y_test,predicted,output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(y_test,predicted,labels=model.classes_).tolist(),'note':'Synthetic shape dataset; benchmark score does not generalize to real-world objects.'}
        joblib.dump({'model':model,'feature_size':extractor.size,'classes':list(model.classes_)},args.model)
        args.report.parent.mkdir(parents=True,exist_ok=True); args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(f'[OK] Trained HOG+SVM on {len(y_train)} images; held-out accuracy={accuracy:.3f}.')
        print(f'[OK] Model: {args.model.resolve()}'); print(f'[OK] Metrics: {args.report.resolve()}')
    else:
        if not args.image: raise SystemExit('ERROR: --image is required when --mode predict.')
        if not args.image.is_file(): raise SystemExit(f'ERROR: Input image not found: {args.image}')
        if not args.model.is_file(): raise SystemExit(f'ERROR: Model not found: {args.model}. Run --mode train first.')
        try: bundle=joblib.load(args.model)  # only load artifacts produced by a trusted local run
        except Exception as exc: raise SystemExit(f'ERROR: Could not load model: {exc}') from exc
        image=cv2.imread(str(args.image))
        if image is None: raise SystemExit(f'ERROR: Could not read image: {args.image}')
        vector=extractor.extract(image).reshape(1,-1); pred=bundle['model'].predict(vector)[0]
        prob=bundle['model'].predict_proba(vector)[0]; confidence=float(np.max(prob))
        print(f'[PREDICTION] {pred} | model probability={confidence:.3f}')
if __name__=='__main__': main()
