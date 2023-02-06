import cv2
import matplotlib.pyplot as plt
import numpy as np

from .detection import Detection
from .generate_detections import create_box_encoder
from .nn_matching import NearestNeighborDistanceMetric
from .preprocessing import non_max_suppression
from .track import Track
from .tracker import Tracker


class DeepSORTTracker:
    def __init__(self, reid_model, cosine_thresh, nn_budget, nms_max_overlap):
        self.encoder = create_box_encoder(reid_model, batch_size=1)
        self.tracker = Tracker(
            NearestNeighborDistanceMetric("cosine", cosine_thresh, nn_budget)
        )
        self.nms_max_overlap = nms_max_overlap

    def track(self, frame, bboxes, scores, classes):
        feats = self.encoder(frame, bboxes)
        indices = non_max_suppression(bboxes, self.nms_max_overlap, scores)
        dets = [Detection(*args) for args in zip(bboxes, scores, feats, classes)]
        dets = [dets[i] for i in indices]

        self.tracker.predict()
        self.tracker.update(dets)

        color_map = plt.get_cmap("tab20b")(np.linspace(0, 1, 20))[:, :3] * 255
        track: Track
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.class_name
            color = color_map[int(track.track_id) % 20]
            cv2.rectangle(
                frame,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color,
                2,
            )
            cv2.rectangle(
                frame,
                (int(bbox[0]), int(bbox[1] - 30)),
                (
                    int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17,
                    int(bbox[1]),
                ),
                color,
                -1,
            )
            cv2.putText(
                frame,
                class_name + " : " + str(track.track_id),
                (int(bbox[0]), int(bbox[1] - 11)),
                0,
                0.6,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )
