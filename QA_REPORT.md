# QA Report — OpenCV Professional Project Suite

## Automated tests

- Unit tests: **10 passed**.
- Python syntax/bytecode compilation: completed successfully for the suite at build time.
- Image algorithms covered by tests: thresholding, contour extraction, HSV color detection, pixel-based object measurement, QR URL allowlist, perspective transform, GrabCut, watershed, HOG features, and geometric shape features.

## Offline smoke tests completed

- `threshold.py` and `contours.py`: generated annotated/mask outputs from bundled still images.
- `centroid_trajectory.py`: processed all 100 frames from the packaged AVI.
- `dense_optical_flow.py`, `sparse_optical_flow.py`, and `motion_detection.py`: processed 20 frames each in no-display mode.
- `object_tracker.py`: MIL tracker initialized from an explicit ROI and processed the 100-frame AVI.
- `color_detector.py`: detected one red, one green, and one blue region in the bundled scene.
- `face_eye_detection.py`: detected one face and two eye regions in the included portrait on the build machine.
- `object_measurement.py`: found four contours above its sample threshold in the included scene.
- `qr_scanner.py`: decoded the sample HTTP(S) payload and did not open a browser by default.
- Phase3 image pipelines: GrabCut, perspective correction, demo undistortion, and watershed generated outputs.
- `camera_calibration.py`: detected 54 chessboard inner corners in all 12 synthetic smoke-test views and wrote `demo_only_calibration.npz`.
- Phase4 classifiers: both models trained on 72 generated training samples, evaluated on 18 generated held-out samples, produced JSON reports, and predicted `triangle` for the bundled sample image.

## Not verified in this environment

- Real webcam, microphone, or Windows media-key behavior.
- Runtime installation of MediaPipe and its model pipeline: MediaPipe was not installed in the build environment, so `Drowsiness Detector` and `Hand Gesture Volume Control` were syntax-checked but not executed.
- Spoken alert playback: requires a working local `pyttsx3`/OS TTS engine.
- Real camera calibration accuracy: generated checkerboard views prove only that the calibration pipeline runs; use 10–20 varied photos of a physical checkerboard for a real calibration.
- Classifier generalization outside the synthetic circle/square/triangle dataset. The reported 100% scores are limited to this generated toy dataset and must not be presented as real-world performance.

## Release notes

This bundle includes source, per-project requirements, per-project README files, synthetic test assets, and selected generated output examples. It does not include a virtual environment or any credentials. Projects are separate entry points and do not launch one another.
