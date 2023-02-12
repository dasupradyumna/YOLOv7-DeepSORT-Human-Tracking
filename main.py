import argparse
from os import path
from time import perf_counter

import cv2
import numpy as np

from api.deepsort import DeepSORTTracker
from api.yolo import YOLOPersonDetector

# constants
YOLO_MODEL = "./checkpoints/yolov7x.pt"
REID_MODEL = "./checkpoints/ReID.pb"
MAX_COS_DIST = 0.5
MAX_TRACK_AGE = 100


def video_writer_same_codec(video: cv2.VideoCapture, save_path: str) -> cv2.VideoWriter:
    """
    This function returns a VideoWriter object with the same codec as the input VideoCapture object
    """
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*"avc1")
    return cv2.VideoWriter(save_path, codec, fps, (w, h))


def track_people(input_vid: str, save_path: str):
    """
    Main function which implements the pipeline:
     1. Reads images from an input video stream
     2. Get detections of people in the input frame using YOLO
     3. Processes the detections along with previous tracks using DeepSORT
     4. Each output frame with refined bounding boxes to an output video stream
    """
    global YOLO_MODEL, REID_MODEL, MAX_COS_DIST, MAX_TRACK_AGE

    # initialize Yolo person detector and DeepSORT tracker
    detector = YOLOPersonDetector()
    detector.load(YOLO_MODEL)
    tracker = DeepSORTTracker(REID_MODEL, MAX_COS_DIST, MAX_TRACK_AGE)

    # initialize video stream objects
    video = cv2.VideoCapture(input_vid)
    output = video_writer_same_codec(video, save_path)

    # core processing loop
    frame_i = 0
    time_taken = 0
    while True:
        start = perf_counter()

        # read input video frame
        ret, frame = video.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # process YOLO detections
        detections = detector.detect(frame)
        try:
            bboxes, scores, _ = np.hsplit(detections, [4, 5])
            bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, :2]
            n_objects = detections.shape[0]
        except ValueError:
            bboxes = np.empty(0)
            scores = np.empty(0)
            n_objects = 0

        # track targets by refining with DeepSORT
        tracker.track(frame, bboxes, scores.flatten())

        # write to output video
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        output.write(frame)

        # calculate FPS and display output frame
        frame_time = perf_counter() - start
        cv2.imshow("Detections", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("<< User has terminated the process >>")
            break
        time_taken += frame_time
        frame_i += 1
        print(
            f"Frame {frame_i}: "
            f"{n_objects} people - {int(frame_time*1000)} ms = {1/frame_time:.2f} Hz"
        )

    # print performance metrics
    print(
        f"\nTotal frames processed: {frame_i}"
        f"\nVideo processing time: {time_taken:.2f} s"
        f"\nAverage FPS: {frame_i/time_taken:.2f} Hz"
    )
    cv2.destroyWindow("Detections")


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

    # main pipeline
    track_people(args.input_vid, args.save_path)

    print(f"Total time: {perf_counter()-start:.2f} s")
