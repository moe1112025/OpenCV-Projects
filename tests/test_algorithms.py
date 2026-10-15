from __future__ import annotations
import importlib.util
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def load_module(name, relative_path):
    path=ROOT/relative_path
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module

def test_threshold_returns_single_channel_image():
    module=load_module('threshold_demo','Beginner Projects CLI/threshold.py')
    image=cv2.imread(str(ROOT/'Beginner Projects CLI/test_images/threshold_scene.png'))
    result=module.process(image,mode='otsu')
    assert result.shape==image.shape[:2]
    assert set(np.unique(result)).issubset({0,255})

def test_contour_demo_finds_shapes():
    module=load_module('contour_demo','Beginner Projects CLI/contours.py')
    image=cv2.imread(str(ROOT/'Beginner Projects CLI/test_images/geometric_scene.png'))
    output,count,mask=module.process(image,min_area=600)
    assert count>=3
    assert output.shape==image.shape and mask.shape==image.shape[:2]

def test_color_detector_handles_test_scene():
    module=load_module('color_demo','Color Detector/color_detector.py')
    detector=module.ColorDetector(min_area=400)
    image=cv2.imread(str(ROOT/'Color Detector/test_images/color_scene.png'))
    output,summary=detector.process(image)
    assert output.shape==image.shape
    assert summary['RED']>=1 and summary['GREEN']>=1 and summary['BLUE']>=1

def test_object_measurement_reports_pixel_units_by_default():
    module=load_module('measurement_demo','Object Measurement/object_measurement.py')
    image=cv2.imread(str(ROOT/'Object Measurement/test_images/object_scene.png'))
    output,items,mask=module.ObjectMeasurement(min_area=800).process_frame(image)
    assert output.shape==image.shape and mask.shape==image.shape[:2]
    assert len(items)>=2
    assert 'width_px' in items[0]

def test_qr_url_policy_blocks_non_http_payloads():
    module=load_module('qr_demo','QR Code Scanner/qr_scanner.py')
    assert module.safe_http_url('https://example.com/path')
    assert module.safe_http_url('http://example.com')
    assert not module.safe_http_url('file:///C:/Windows/win.ini')
    assert not module.safe_http_url('javascript:alert(1)')
    assert not module.safe_http_url('example.com')

def test_perspective_transform_produces_rectified_image():
    module=load_module('perspective_demo','Phase3/perspective_transform.py')
    image=cv2.imread(str(ROOT/'Phase3/test_images/document_scene.png'))
    result,points=module.transform(image)
    assert result.ndim==3 and result.shape[0]>100 and result.shape[1]>100
    assert points.shape==(4,2)

def test_grabcut_mask_has_foreground_pixels():
    module=load_module('grabcut_demo','Phase3/grabcut_segmentation.py')
    image=cv2.imread(str(ROOT/'Phase3/test_images/grabcut_scene.png'))
    result,mask=module.process(image,iterations=2)
    assert result.shape==image.shape and mask.shape==image.shape[:2]
    assert cv2.countNonZero(mask)>0

def test_watershed_returns_label_map():
    module=load_module('watershed_demo','Phase3/watershed_segmentation.py')
    image=cv2.imread(str(ROOT/'Phase3/test_images/watershed_scene.png'))
    result,markers,mask=module.process(image)
    assert result.shape==image.shape and markers.shape==image.shape[:2]
    assert mask.shape==image.shape[:2]
    assert np.any(markers==-1)

def test_hog_extractor_returns_finite_vector():
    module=load_module('hog_demo','Phase4/hog_svm_classifier.py')
    image=cv2.imread(str(ROOT/'Phase4/test_images/sample_shape.png'))
    vector=module.HOGFeatureExtractor().extract(image)
    assert vector.ndim==1 and vector.size>0 and np.isfinite(vector).all()

def test_geometry_features_are_finite():
    module=load_module('geometry_classifier_demo','Phase4/image_classification.py')
    image=cv2.imread(str(ROOT/'Phase4/test_images/sample_shape.png'))
    vector=module.extract_features(image)
    assert vector.ndim==1 and vector.size>=10 and np.isfinite(vector).all()

