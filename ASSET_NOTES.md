# Asset and Test-Data Notes

- Images with geometric shapes, colors, a document, segmentation objects, a grid, checkerboards, QR, and the motion AVI were generated locally for this suite. They are synthetic QA fixtures, not a real-world benchmark.
- `Face and Eye Detection/test_images/face_reference.png` and `Drowsiness Detector/test_images/face_reference.png` are the `astronaut` sample image shipped by scikit-image at build time. It is included only to demonstrate reading and processing a face-containing image. For provenance or reuse outside this study project, review the scikit-image sample-data documentation and applicable source-image terms.
- Haar cascade XML files are copied from the OpenCV Python package distribution's `cv2.data.haarcascades` directory and are included beside the Face and Eye Detection script for portability.
- The generated checkerboard views are intended for testing the calibration code path only. Do not use them to infer real camera calibration accuracy.
- The shape dataset contains three generated classes (`circle`, `square`, `triangle`) with predefined `train` and `test` splits. Reports are demonstration metrics and should not be presented as performance on external images.
