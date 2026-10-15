# Beginner Projects Vision

Standalone demonstrations for centroid trajectory, dense optical flow, sparse optical flow, motion detection, and ROI-based object tracking. None of these scripts import or invoke the others.

## Install

```powershell
python -m pip install -r requirements.txt
```

## Run the included motion sample

```powershell
python centroid_trajectory.py
python dense_optical_flow.py
python sparse_optical_flow.py
python motion_detection.py
python object_tracker.py
```

The tracker opens an ROI picker; drag a box around the moving circle and press **Enter**. If the default CSRT tracker is unavailable, try `python object_tracker.py --tracker MIL`.

## Use a webcam or another file

```powershell
python motion_detection.py --source 0
python sparse_optical_flow.py --source "C:\videos\clip.mp4"
python object_tracker.py --source 0 --tracker KCF
```

For a headless smoke run (no windows):

```powershell
python dense_optical_flow.py --no-display --max-frames 30
python motion_detection.py --no-display --max-frames 30
```

## Sample data

`test_images/sample_motion.avi` is a generated synthetic clip, suitable for checking that a video pipeline runs. It is not a benchmark dataset and does not represent real-world motion complexity.
