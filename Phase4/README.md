# Phase4 — Classical Image Classification

Two independent classification experiments run on a small generated **circle / square / triangle** dataset. They demonstrate two different feature approaches, not a general-purpose object-recognition product:

1. `hog_svm_classifier.py` uses HOG descriptors with an RBF SVM.
2. `image_classification.py` uses contour geometry/Hu moments with a Random Forest.

## Install and train

```powershell
python -m pip install -r requirements.txt
python hog_svm_classifier.py --mode train
python image_classification.py --mode train
```

Both scripts evaluate a held-out `test` split and write a local model file plus JSON metrics under `outputs/`.

## Predict a sample image

```powershell
python hog_svm_classifier.py --mode predict --image test_images\sample_shape.png
python image_classification.py --mode predict --image test_images\sample_shape.png
```

To train on your own images, place class folders under `--dataset` using `train/class_name/` and `test/class_name/`. Both splits must contain the same classes. Do not compare the synthetic dataset score with performance on photographs: these samples contain simple, centered geometric shapes only.

**Model security:** both scripts use `joblib` to load the model they created locally. Only load trusted model files. Pickle-based artifacts can execute code during deserialization if malicious.
