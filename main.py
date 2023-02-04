import argparse

from yolo.api import YoloDetector

# hyperparameters
YOLO_MODEL = "./models/yolov7x.pt"
REID_MODEL = "./models/ReID.pb"
MAX_COS_DIST = 0.4
NN_BUDGET = ...
NMS_MAX_OVERLAP = 1


def track_people(input_vid: str, save_path: str):
    global YOLO_MODEL, REID_MODEL, MAX_COS_DIST, NN_BUDGET, NMS_MAX_OVERLAP
    # initialize Yolo detector (with weights)
    yolo = YoloDetector(classes=[0])
    yolo.load(YOLO_MODEL)
    # initialize DeepSORT tracker
    # core loop
    ## read input video
    ## get detections
    ## SORT tracking
    ## write to output video
    ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Track and ID People in a video",
        description="Use Yolov7 for detecting people in a video, assign IDs to detected"
        " people and track them as long as they are visible",
    )
    parser.add_argument(
        "--input-vid",
        type=str,
        default="./data/input.mp4",
        help="path to the input video file to track people",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./data/output.avi",  # TODO change to MP4
        help="path to save file the output video",
    )

    args = parser.parse_args()
    track_people(args.input_vid, args.save_path)
