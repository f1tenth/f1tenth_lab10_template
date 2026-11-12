# Lab 10: Vision Lab

Welcome to the Vision Lab. This lab has four parts. Parts 1 through 3 are straightforward. Part 4 is more involved, so plan your time accordingly. Please work in teams and make full use of online resources. Many developers have implemented these methods before. If you run into a problem, someone has most likely already answered it on a developer forum such as Stack Overflow.

## I. Learning Goals

- Use of v4l2 to access and control camera device.
- Calibration of camera intrinsics and extrinsics with other sensors (e.g. Lidar, IMU)
- Triangulating the target object in 3D using depth
- Training and Deployment of an object detection with TensorRT

## II. Use cameras on Linux system
As discussed in class, cameras in Linux are device files under `/dev`. Connect the RealSense camera you received to the Jetson and run `ls /dev/video*`. You should see three video devices. Use OpenCV to capture RGB frames from `/dev/video2`.

During racing, we want to receive camera frames as fast as possible. The highest RGB resolution at which the RealSense camera runs at 60 Hz is 960x540. Modify your code to receive 960x540 images at 60 Hz.

<!-- Then write a ROS 2 node that captures RGB frames from the RealSense camera and publishes them on the `/rgb_img` topic using the `sensor_msgs/Image` message type. -->

Save this part of the code as `camera.py` or `camera.cpp`.

## III. Camera Calibration
For this course, we model the camera as a pinhole camera. The intrinsic parameters of the camera are its focal lengths and its principal point. We use them to map points in front of the camera to pixels on the image plane. A point $P_C = (X, Y, Z)$ in the camera frame, with $Z > 0$ in front of the camera, projects to the pixel $(u, v)$ given by

$$
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \frac{1}{Z} K \begin{bmatrix} X \\ Y \\ Z \end{bmatrix},
\qquad
K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
\quad\Longleftrightarrow\quad
u = f_x \frac{X}{Z} + c_x, \quad v = f_y \frac{Y}{Z} + c_y.
$$

Here $f_x$ and $f_y$ are the focal lengths in pixels. The principal point $(c_x, c_y)$ is the pixel at the intersection of the optical axis and the image plane. Read these values from the `camera_info` topic of the RealSense color stream. **Each resolution has its own $K$.** If you resize the image by a factor $s$, multiply $f_x$, $f_y$, $c_x$ and $c_y$ by $s$ as well. If all distortion coefficients in `camera_info` are zero, you can skip undistortion. This is usually the case for the RealSense color stream.

Since the projection divides by $Z$, we cannot recover depth from a single pixel. We can only recover the ray through the camera center and that pixel,

$$
d_C = K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \begin{bmatrix} (u - c_x)/f_x \\ (v - c_y)/f_y \\ 1 \end{bmatrix},
\qquad P_C = Z d_C \quad \text{for an unknown depth } Z.
$$

If we know the depth $Z$ of the pixel from the RealSense depth image or from the LiDAR, then the 3D point is

$$
P_C = Z d_C = \begin{bmatrix} (u - c_x) Z / f_x \\ (v - c_y) Z / f_y \\ Z \end{bmatrix}.
$$

See [Pinhole Camera Model](https://hedivision.github.io/Pinhole.html) for more detail.

### Optional
Camera calibration can change slightly over a period of time and might not match the manufacturer's intrinsic calibration. One can recalibrate the intrinsics of the camera using OpenCV checkerboard [tutorial](https://docs.opencv.org/4.13.0/dc/dbb/tutorial_py_calibration.html). For multi camera system or Camera-IMU extrinsic calibration one can use [Kalibr](https://github.com/ethz-asl/kalibr). 

## IV. Distance Measurement

In this part you will understand camera intrinsic and extrinsic calibration. Intel RealSense has a stereo camera setup, and IR projector based depth estimation. RealSense D435 has a depth range of 0.6m - 6m. Read more in Intel documents. \

Given, pixel (x, y), camera intrinsics, and corresponding depth one can project the point in 3D with respect to the camera position. To get accurate long range depth, use LiDAR depth values. LiDAR is fixed with respect to the camera.

### A. 2D to 3D projection  

Every 3D point on a ray through the lens center lands on the same pixel. The depth value can be obtained from 2D lidar scan but it is on a different coordinate frame than camera coordinate frame. Thus, convert the depth scan to camera coordinate frame and use that to get the position of object with respect to the camera.


#### 1. Coordinate frames

| Frame | Axes | Notes |
| --- | --- | --- |
| LiDAR $L$ | $x$ forward, $y$ left, $z$ up | The scan is in the plane $z_L = 0$. |
| Camera optical $C$ | $x$ right, $y$ down, $z$ forward | OpenCV convention. The ROS name is `*_optical_frame`. |

#### 2. Intrinsics: camera frame → image

We use the pinhole model with the intrinsic matrix $K$ of the color camera. Use the $K$ that matches the resolution of your images. We project a point $P_C = (X_C, Y_C, Z_C)$ with $Z_C > 0$ to the pixel $(u, v)$ with

$$
u = f_x \frac{X_C}{Z_C} + c_x, \qquad v = f_y \frac{Y_C}{Z_C} + c_y.
$$

Conversely, if we know the depth $Z_C$ of a pixel $(u, v)$, then we back-project the pixel to the point

$$
P_C = Z_C K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \begin{bmatrix} (u - c_x) Z_C / f_x \\ (v - c_y) Z_C / f_y \\ Z_C \end{bmatrix}.
$$

Here $Z_C$ is the coordinate of the point along the optical axis. Do not use the LiDAR range $r_i$ in place of $Z_C$, since $r_i$ is the distance from the LiDAR to the point. We use the projection in Section 4 and the back-projection in Section 5.

#### 3. Extrinsics: LiDAR frame → camera frame

Since the LiDAR and the camera are rigidly mounted, we relate their frames with a rigid transformation

$$
P_C = R_{CL} P_L + t_{CL}.
$$

The rotation $R_{CL}$ is the fixed **axis swap** from the forward-left-up axes of the LiDAR to the right-down-forward axes of the camera, with $X_C = -y_L$, $Y_C = -z_L$ and $Z_C = x_L$. With this rotation, we assume that the axes of the two sensors are aligned. Mount both sensors rigidly with aligned axes to avoid small residual rotations.

$$
R_{CL} = \begin{bmatrix} 0 & -1 & 0 \\ 0 & 0 & -1 \\ 1 & 0 & 0 \end{bmatrix},
\qquad t_{CL} = -R_{CL} c_L
$$

Here $c_L$ is the position of the camera lens center in the LiDAR frame. For example, if we mount the camera 7.6 cm in front of the LiDAR and 6.35 cm below it, then $c_L = (0.076, 0, -0.0635)$ m and $t_{CL} = (0, -0.0635, -0.076)$ m. Measure $c_L$ to the lens of the color camera, since we project the scan into the color image. The color camera of the D435 is not at the center of the case, so be sure to get the transformation to the color camera's lens.

#### 4. Projecting the LiDAR scan into the image

For beam $i$ of a `sensor_msgs/LaserScan`, with range $r_i$ and angle $\theta_i = \theta_{min} + i \Delta\theta$, the point in the LiDAR frame is

$$
P_{L,i} = (r_i \cos\theta_i, r_i \sin\theta_i, 0).
$$

Transform each point to the camera frame with the extrinsics. Discard points with $Z_C \le 0$. Project the remaining points with $K$ and plot them on the image. **This plot is your calibration check.** If your extrinsics are correct, you should see the projected scan on objects at the height of the LiDAR. You should also see the range jumps of the scan at the edges of objects. If you see a horizontal shift, then the lateral offset in your extrinsics is wrong. If you see a vertical shift, then the height offset is wrong.

<!-- Since the camera center is not in the scan plane, the projected scan is a curve. If we mount the camera with no tilt and the LiDAR at a height $h$ above the lens, then we find a scan point at depth $Z_C$ on the row

$$
v = c_y - f_y \frac{h}{Z_C}.
$$

Hence the scan points of near objects are higher in the image. As $Z_C \to \infty$, we have $v \to c_y$. Hence the scan points of far objects are close to the horizon row $v = c_y$.

#### 5. Pixel or bounding box to 3D with the LiDAR

Since we only have LiDAR depth along this curve, we take the depth of a pixel $(u, v)$ from the LiDAR returns **in its image column**.

1. **Gate.** Keep the projected scan points with $|u_i - u| < w$ for a small window half-width $w$. For a bounding box, use the columns of the box. Trim about 15% from each side of the box to exclude the background at its edges.
2. **Cluster and pick the nearest.** Split the gated returns into clusters wherever the range jump between neighboring beams is more than about $3\sigma$ above the expected jump. Here $\sigma$ is the range noise of the LiDAR. Keep the **closest** cluster, since we cannot see anything behind the object. Interpolate this cluster at column $u$ to get the LiDAR hit $h_L = (h_x, h_y, 0)$.
3. **Intersect the pixel ray with the surface of the object.** Assume the surface is vertical at $h_L$. We can make this assumption for cars, walls and cones. Let $a = -R_{CL}^{\top} t_{CL}$ be the camera center in the LiDAR frame. Let $d = R_{CL}^{\top} K^{-1} [u, v, 1]^{\top}$ be the direction of the pixel ray in the LiDAR frame. We take the vertical plane through $h_L$ with the horizontal normal $n = (h_x - a_x, h_y - a_y, 0) / \lVert\cdot\rVert$. Then we find the intersection point $P_L$ as
$$
\lambda = \frac{n^{\top} (h_L - a)}{n^{\top} d}, \qquad P_L = a + \lambda d .
$$
4. **Transform** $P_L$ to the frame you need. For example, use the mounting pose of the LiDAR to transform $P_L$ to `base_link`.

For a level camera, an image column is a vertical plane through the lens. Hence step 3 is exact for any vertical surface, including at pixels above or below the scan line. A simpler alternative is to use the depth of the hit, $Z_C = (R_{CL} h_L + t_{CL})_z$, in the back-projection formula of Section 2. Since $R_{CL}$ is a pure axis swap, the camera is level relative to the LiDAR plane. Then we obtain the same point from this shortcut and from step 3. We get different points from the two methods only when the camera is tilted relative to the LiDAR.

**Resolution.** If you run the detector on a resized image of size $W_{det} \times H_{det}$, scale the box back to the camera resolution $W \times H$ with $u = u_{det} W / W_{det}$ and $v = v_{det} H / H_{det}$ before using $K$.

**Limitations to keep in mind.**

- If an object is entirely below or above the LiDAR plane, we assign it the depth of the background behind it. Check that the projected LiDAR hit is inside the box vertically. Otherwise, reject the box.
- If a small box is next to a nearer object, we may pick up the returns of the nearer object.
- At short range, the RealSense depth image is a useful cross-check. The D435 depth is reliable up to about 3 m. See the optional part of Section VI. -->

## V. Object Detection Network Training & Deployment

In this part, you will first train an object detection nerual network and deploy it onto the F110 car. The model you will use is RT-DETRv2 which is the real-time detection model with transformers developed by Facebook. The particular version you will implement has 80 classes. First infer it directly on the dataset image and observe the output classes. Next you can either add `f1tenth_car` as an additional class or only keep this class as the output. By default it keeps all classes, thus remove other object detections which do not belong to your class of interest. The purpose of this assignment is to understand using open source model architecture from source code and finetune or modify the architecture as required.  

### Neural Network Training

#### Upload to Google Colab

For the network training, we provide labeled dataset and a simple YOLO training and testing notebook written in Pytorch. You can access them [here](https://drive.google.com/drive/folders/18gjExE2XiQBj1f0_pgRPvbCSnguI6cLS?usp=sharing). You can use it to train on Google Colab with the free GPU resource. If you are new to Pytorch, watch this [tutorial](https://www.youtube.com/watch?v=Jy4wM2X21u0) as it goes through some key components in a training pipeline.

You can create an ese_615 folder in the 'My Drive' in your Google Drive. Upload the notebook and unzipped dataset folder. Open the note book with Google Colab and make sure you are connected to a GPU instance. The notebook has everything you need to for training and testing. For some of post-processing functions, you can just borrow them when you do you deployment.

#### TODOs

There are only a few positions where marked 'TODO' or 'STUDENT CODE HERE'. The model arhcitecture is given in the notebook, it has a ResNet backbone, transformer encoder and decoder. You have to fill out parts of the encoder definition. Decoder has a more complicated deformable attention model and therefore it's code is provided. \
The next 'TODO' are some hyperparameters: batchsize, learning rate and epoch. Large batchsizes make you learn faster, but lower batchsizes give more randomness. Large learning rate make you learn faster but may be unstable (producing NaNs) and bounce around the optimal point. Epoch is just how many times you run with all the dataset. Play around with this values and get a sense of what's suitable. Since our detection task is simple and the dataset is not diverse. You should be able to drop the training error below 1000 quite easily and you won't need to pay for a fast GPU to complete this lab. We are not grading on the accuracy, as long as it can detect somewhat. \
Some of the code is hidden blocks. If interested read through the code to understand the entire pipeline. 


#### Use Other Networks

If you are familiar with network training, you can choose any network architecture you like and train it with our dataset. You can receive great **extra credit** if you train a better object detection NN that is more robust. You need to make sure that it can be successfully deployed with TensorRT 8.5, which is the latest version you can get for Jetson. List your name in the `submission.md` if you did this.

### TensorRT Deployment

After you are done with training the network, convert it into a TensorRT engine on the Jetson NX. To do this, first convert the model from PyTorch into ONNX and then to TensorRT. Then write a function that takes in an image, preprocess it, run with the model, post-process and return the results. Refer to the sample [code](https://github.com/NVIDIA/TensorRT/tree/release/8.4/samples) provided by TensorRT for building the engine. Make sure you select the correct branch for your TensorRT version. Some other references are attached at the end. **Notice they may be for a previous version.**

Save this part of the code as 'convert_trt.py' or 'convert_trt.cpp' and 'detection.py' or 'detection.cpp'.

#### Install Pytorch
If your Jetson doesn't have Pytorch, follow [this](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html) to install Pytorch.

#### Install ONNX and matplotlib
```
sudo apt-get install python3-matplotlib
sudo apt-get install python3-pip libprotoc-dev protobuf-compiler
pip3 install onnx
```

#### Install pycuda and set up TensorRT
If your Jetson doesn't have pycuda, follow [this](https://docs.donkeycar.com/guide/robot_sbc/tensorrt_jetson_nano/) to set up TensorRT and install pycuda. (Only do Step 2)

#### Implementation Tips

1. In your deployment order, do not import Pytorch. Use numpy to reshape the output to (5, 5, 10) if you use my code.

2. Convert the the engine using FP32 and FP16 mode and compare the speed difference. See `submission.md`.
 
## VI. Combining the Functions

In the final part, write a complete package that takes image from the camera, run object detection on the image, then use the center of the bounding box that falls on the 2D LiDAR projection line to project to 3D world position and get object distance from camera. Please note that the image resolution should be 960x540 and your detection resolution may be different. \n
Save an image that shows the bounding box output for the detected class `f1tenth_car` with the 3D projection value. Name the image `detection`. Take a picture showing the distance between camera and the car with a measuring tape. Submit both images. Name the image `measurement`. 


Save this part of the code as 'integrated.py' or 'integrated.cpp'.

### (Optional)
As you observe the 3D distance of the object you will notice that using only one point for 3D projection is very fluctuating. One can average on multiple observations to get a good estimate of the 3D position. Alternatively, use the depth value from camera, which allows you to sample multiple points everywhere on the bounding box and observe if there is a difference in the projected value from the LiDAR depth as the object moves farther away. Empirically, which one gives better estimate?  \
Utilise these observations for your future projects. To get more robust predictions of the distance of the car, smooth out the predicted distance value in time using averaging methods, and filter out outliers. 



<!-- In the final part, we will write another ROS node that subscribes to `/rgb_img` channel for an image. Run the lane detection and object detection on the image. Then use the center point of the botton bounding box edge from object detection to calcuate the distance to the object. Please note that the image resolution should be 960x540 and your detection resolution is smaller. -->

<!-- Publish the result image with lane detection and object detection indicators to the `/det_img` channel. Publish the distance to detections to the `/det_distance` channel. -->

## VII. Deliverables and Submission

Push your finished code onto github repo. Make you name the parts as instructed.

Please follow the `submission.md` in the Github repo.

## VIII: Grading Rubric

- Accessing Image: **10** Points
- Distance Measurement: **25** Points
- Network Training: **15** Points
- TensorRT Deployment: **40** Points
- Integration: **10** Points

## Other References
1. [tutorial](https://learnopencv.com/how-to-convert-a-model-from-pytorch-to-tensorrt-and-speed-up-inference/)
2. [guide](https://docs.nvidia.com/deeplearning/tensorrt/sample-support-guide/index.html)
3. You can see [this](https://github.com/NVIDIA-AI-IOT/torch2trt/issues/233) on how to save and read engine file. Also, due to version difference, we need to replace this [line](https://github.com/spmallick/learnopencv/blob/a18fa4e1a255f58700b3c4687e425cabd58c41bf/PyTorch-ONNX-TensorRT/trt_inference.py#L17) by the following two lines.
```
explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(explicit_batch)
```
