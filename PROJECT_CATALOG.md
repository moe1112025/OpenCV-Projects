# Project Catalog

| Project folder | Entry point | Main technique | Included sample input | Typical output |
|---|---|---|---|---|
| Beginner Projects CLI | `threshold.py` | Fixed/Otsu thresholding | `test_images/threshold_scene.png` | Binary mask PNG |
| Beginner Projects CLI | `contours.py` | Contour detection / bounding boxes | `test_images/geometric_scene.png` | Annotated image + mask |
| Beginner Projects Vision | `centroid_trajectory.py` | Background subtraction + centroids | `test_images/sample_motion.avi` | Live trajectory window |
| Beginner Projects Vision | `dense_optical_flow.py` | Farneback dense flow | `test_images/sample_motion.avi` | HSV flow visualization |
| Beginner Projects Vision | `sparse_optical_flow.py` | Lucas–Kanade / Shi–Tomasi | `test_images/sample_motion.avi` | Feature tracks |
| Beginner Projects Vision | `motion_detection.py` | MOG2 background model | `test_images/sample_motion.avi` | Motion regions/mask |
| Beginner Projects Vision | `object_tracker.py` | CSRT/KCF/MIL ROI tracking | `test_images/sample_motion.avi` | Bounding box window |
| Color Detector | `color_detector.py` | HSV ranges + morphology | `test_images/color_scene.png` | Color boxes/labels |
| Drowsiness Detector | `drowsiness_detector.py` | MediaPipe Face Mesh + EAR | Webcam / portrait fixture | Eye contours + temporal status |
| Face and Eye Detection | `face_eye_detection.py` | Haar cascade classifiers | `test_images/face_reference.png` | Face/eye rectangles |
| Object Measurement | `object_measurement.py` | Contours and scale conversion | `test_images/object_scene.png` | Dimensions/area + mask |
| QR Code Scanner | `qr_scanner.py` | OpenCV QR detector | `test_images/sample_qr.png` | Decoded payload + overlay |
| Hand Gesture Volume Control | `hand_gesture_volume_control.py` | MediaPipe landmarks + pinch ratio | Webcam | One volume key on state change |
| Phase3 | `camera_calibration.py` | Chessboard corners + `calibrateCamera` | `test_images/calibration_samples/` | NPZ calibration file |
| Phase3 | `grabcut_segmentation.py` | GrabCut foreground mask | `test_images/grabcut_scene.png` | Segmented PNG + mask |
| Phase3 | `perspective_transform.py` | Quadrilateral detection + homography | `test_images/document_scene.png` | Rectified document |
| Phase3 | `undistort_image.py` | `cv2.undistort` | `test_images/lens_grid.png` | Undistorted PNG |
| Phase3 | `watershed_segmentation.py` | Distance transform + watershed | `test_images/watershed_scene.png` | Watershed boundaries |
| Phase4 | `hog_svm_classifier.py` | HOG + SVM | `test_images/shapes_dataset/` | Model + metrics JSON |
| Phase4 | `image_classification.py` | Shape geometry + Random Forest | `test_images/shapes_dataset/` | Model + metrics JSON |

Folder names are kept close to your screenshot. The implementation in each folder is deliberately independent; common patterns are duplicated rather than centralized into a shared runtime package, so you can study, break, and improve each file separately.
