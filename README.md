# Tracking People using YOLOv7 and DeepSORT

This repository implements a solution to the problem of tracking moving people in a low-quality
video. It uses a state-of-the-art object detector **YOLOv7** for detecting people in a frame, and
fuses these robust detections with the bounding boxes of previously tracked people using the neural
network version of SORT called **DeepSORT** tracker. The packages for **YOLO** and **DeepSORT**
algorithms are located under `yolo` and `deepsort` folders, where the modules are adapted from the
official repositories to fit the scripts here.

## Table of Contents

1. [Get Started](#get-started)
2. [Code](#code)
3. [Results](#results)
4. [External Work](#external-work)

## Get Started

The necessary python environment required for working with this repository can be setup using the
script `setup_conda.sh`. This script assumes that the system has a working **conda** distribution
(Miniconda or Anaconda), and accepts one argument which will be used as the name for the newly
created environment.  
The conda environment will have the below major packages:

1. **TensorFlow** 2.8.1 (with **GPU** support)
2. **CUDA** Toolkit 10.2 (with **CuDNN** 7.6.5)
3. **PyTorch** 1.10.1 (with **TorchVision** 0.11.2)
4. **OpenCV** 4.6.0
5. **Matplotib** 3.6.3

**Note**: The script is not robust, it may break when conda packages in their respective channels
are updated.

## Code

After setting up the conda environment using the above script, the pipeline can be executed by -

```bash
python main.py --input-vid=/path/to/input/video/file --save-path=/path/to/output/video/file
```

This will read frames from the input video, detect people using a **YOLO** API, and track people
using a **DeepSORT** API. These APIs can be found in `api` folder. The frames with people detected
and tracked are saved to the specified video file.

### Results

The input video, versioned output videos, checkpoint files and a detailed README file can be found
[here](https://drive.google.com/drive/folders/1R2AENddPC9sIk5Lp8nSDv2vLoeAyy8-L?usp=share_link).
The different version of the output are from changes made to some parameters in the source code.

## External Work

### YOLOv7

[Repository](https://github.com/WongKinYiu/yolov7) - [Paper](https://arxiv.org/abs/2207.02696)

All the modules related to this network can be found under `yolo/`, and the weights can be found
under `checkpoints/yolov7x.pt`. More details can be found in `yolo/__init__.py`.

### DeepSORT

[Repository](https://github.com/nwojke/deep_sort) - [Paper](https://arxiv.org/abs/1703.07402)

All the modules related to this network can be found under `deepsort/`, and the weights can be found
at `checkpoints/ReID.pb`. More details can be found in `deepsort/__init__.py`.
