"""
This package contains (slightly adapted) modules and packages from the official YOLOv7 repository:
    [ https://github.com/WongKinYiu/yolov7 ]
FROM "yolov7/models"
    ( all )
FROM "yolov7/utils"
    - autoanchor.py
    - datasets.py
    - general.py
    - google_utils.py
    - loss.py
    - plots.py
    - torch_utils.py
"""

import sys
from os import path

# add "yolo" directory to PATH (required by YOLO's torch serialized model checkpoint)
sys.path.append(path.dirname(__file__))
