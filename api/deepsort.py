import cv2
import matplotlib.pyplot as plt
import numpy as np

from deepsort.detection import Detection
from deepsort.generate_detections import create_box_encoder
from deepsort.nn_matching import NearestNeighborDistanceMetric
from deepsort.tracker import Tracker


class DeepSORTTracker:
    """
    DeepSORT tracker wrapper, exposing just one method which receives an image and its detections to
    refine them using existing tracks
    """

    def __init__(
        self,
        reid_model: str,
        cosine_thresh: float,
        max_track_age: int,
        nn_budget: int = None,
    ) -> None:
        self.encoder = create_box_encoder(reid_model, batch_size=1)
        self.tracker = Tracker(
            NearestNeighborDistanceMetric("cosine", cosine_thresh, nn_budget),
            max_age=max_track_age,
        )
        self.colors = plt.get_cmap("hsv")(np.linspace(0, 1, 20, False))[:, :3] * 255

    def track(self, frame: np.ndarray, bboxes: np.ndarray, scores: np.ndarray) -> None:
        """
        Accepts an image and its YOLO detections, uses these detections and existing tracks to get a
        final set of bounding boxes, which are then drawn onto the input image
        """
        feats = self.encoder(frame, bboxes)
        dets = [Detection(*args) for args in zip(bboxes, scores, feats)]

        # refine the detections
        self.tracker.predict()
        self.tracker.update(dets)

        # render the final tracked bounding boxes on the input frame
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr().astype(np.int32)
            color = self.colors[track.track_id % 20]
            # draw detection bounding box
            cv2.rectangle(frame, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
            # draw text box for printing ID
            cv2.rectangle(
                frame,
                tuple(bbox[:2]),
                (bbox[0] + (4 + len(str(track.track_id))) * 8, bbox[1] + 20),
                color,
                -1,
            )
            # print ID in the text box
            cv2.putText(
                frame,
                f"ID: {track.track_id}",
                (bbox[0] + 4, bbox[1] + 13),
                cv2.FONT_HERSHEY_DUPLEX,
                0.4,
                (0, 0, 0),
                lineType=cv2.LINE_AA,
            )
