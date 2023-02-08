import argparse
from os import path
from time import perf_counter

import cv2
import numpy as np

from api.deepsort import DeepSORTTracker
from api.yolo import YOLODetector

# hyperparameters
YOLO_MODEL = "./checkpoints/yolov7x.pt"
REID_MODEL = "./checkpoints/ReID.pb"
MAX_COS_DIST = 0.5
NN_BUDGET = None
IOU_THRESH = 0.5
MAX_TRACK_AGE = 100


def video_writer_same_codec(video: cv2.VideoCapture, save_path: str) -> cv2.VideoWriter:
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*"avc1")
    return cv2.VideoWriter(save_path, codec, fps, (w, h))


def track_people(input_vid: str, save_path: str):
    global YOLO_MODEL, REID_MODEL, MAX_COS_DIST, NN_BUDGET, IOU_THRESH
    # initialize Yolo detector (with weights) to detect only person class
    detector = YOLODetector(classes=[0], iou_th=IOU_THRESH)
    detector.load(YOLO_MODEL)
    # initialize DeepSORT tracker
    tracker = DeepSORTTracker(REID_MODEL, MAX_COS_DIST, NN_BUDGET, MAX_TRACK_AGE)
    # initialize video stream objects
    video = cv2.VideoCapture(input_vid)
    output = video_writer_same_codec(video, save_path)
    # core loop
    frame_i = 0
    time_taken = 0
    while True:
        start = perf_counter()
        # read input video
        ret, frame = video.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # get detections
        detections = detector.detect(frame, draw_bboxes=False)
        if detections is None:
            bboxes = []
            scores = []
            classes = []
            n_objects = 0
        else:
            bboxes, scores, classes = np.hsplit(detections, [4, 5])
            bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, :2]
            n_objects = detections.shape[0]
        names = np.array(
            [detector.class_list[int(classes[i])] for i in range(n_objects)]
        )
        # track targets
        tracker.track(frame, bboxes, scores.flatten(), names)
        # write to output video
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        output.write(frame)
        # display
        cv2.imshow("Processed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("<< User has terminated the process >>")
            break
        time_taken += perf_counter() - start
        frame_i += 1

    print(f"FPS: {frame_i/time_taken:.2f} Hz")
    cv2.destroyAllWindows()


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
        default="./data/output.mp4",
        help="path to save file the output video",
    )

    args = parser.parse_args()
    args.input_vid = path.abspath(args.input_vid)
    args.save_path = path.abspath(args.save_path)
    start = perf_counter()
    track_people(args.input_vid, args.save_path)
    print(f"Time taken: {perf_counter()-start:.2f} s")
