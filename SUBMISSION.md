# Lab 10: Vision Lab

Replace every `(FILL ME IN)`. The TAs grade from this file, your code and your video.

For an image, commit it to this repository and embed it here so it shows on this page, for example `![detection](detection.png)`.

## Team
Team members: (FILL ME IN)

## Code
Give the path of each file in this repository.

| Part | File |
| --- | --- |
| Camera (`camera.py` or `camera.cpp`) | (FILL ME IN) |
| TensorRT conversion (`convert_trt.py` or `convert_trt.cpp`) | (FILL ME IN) |
| Detection (`detection.py` or `detection.cpp`) | (FILL ME IN) |
| Integration (`integrated.py` or `integrated.cpp`) | (FILL ME IN) |
| Training notebook (`.ipynb`, saved with its outputs) | (FILL ME IN) |

## 1. Accessing Images (10 points)
Resolution and frame rate you get from the camera: (FILL ME IN)

## 2. Distance Measurement (25 points)
Intrinsics at 960x540 (fx, fy, cx, cy): (FILL ME IN)

Position of the camera lens in the LiDAR frame, c_L = (x, y, z) in meters: (FILL ME IN)

Image with the LiDAR scan projected on it (your calibration check): (FILL ME IN)

## 3. Network Training (15 points)
Epochs, batch size and learning rate: (FILL ME IN)

Final training loss: (FILL ME IN)

Training loss and validation plot from the notebook: (FILL ME IN)

## 4. TensorRT Deployment (40 points)
FP16 inference time (ms): (FILL ME IN)

FP32 inference time (ms): (FILL ME IN)

Time only the inference step. Average the time over at least 100 inference runs.

Is FP16 faster? Why? (FILL ME IN)

## 5. Integration (10 points)
Detection image (bounding box and 3D position drawn on it): (FILL ME IN)

Measurement image (measuring tape between the camera and the car): (FILL ME IN)

Distance from your package (m): (FILL ME IN)

Distance on the measuring tape (m): (FILL ME IN)

Video link: (FILL ME IN)

Make sure that the course staff can view the video.

## Extra Credit (optional, up to 20 points)

### Camera Recalibration (5 points)
Recalibrated intrinsic matrix K: (FILL ME IN)

Reprojection error (pixels): (FILL ME IN)

How does your K compare with the factory K? (FILL ME IN)

### Other Network (10 points)
Architecture: (FILL ME IN)

How did you show that it is more robust than the provided model? (FILL ME IN)

### LiDAR and Depth Image Comparison (5 points)
With which method did you get the better distance estimate? Why? (FILL ME IN)
