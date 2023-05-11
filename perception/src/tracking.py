from typing import List
from perception import Perception
import cv2 as cv
import numpy as np
import numpy.linalg as la

TRACKERS = {
    "csrt": cv.TrackerCSRT_create,
    "kcf": cv.TrackerKCF_create,
    "boosting": cv.TrackerBoosting_create,
    "mil": cv.TrackerMIL_create,
    "tld": cv.TrackerTLD_create,
    "medianflow": cv.TrackerMedianFlow_create,
    "mosse": cv.TrackerMOSSE_create
}

COLORS = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255)
]

class Tracking(Perception):
    def __init__(self, args: List[str]):
        super().__init__(args)
        self.initialized = False
        self.type, *_ = args
        self.m = 2
        self.objs = ["end_effector", "target"]
        self.n = len(self.objs)
        self.trackerss = None


    def initialize_trackers(self, images):
        for i, image in enumerate(images):    
            cv.imshow(f"image{i}", image)
        print("Press i to initialize trackers.")
        key = cv.waitKey(1)
        if key != ord('i'):
            return
        self.trackerss = [cv.MultiTracker_create() for _ in range(self.m)]
        for i, (image, trackers) in enumerate(zip(images, self.trackerss)):
            print(f"Click the top left and then the bottom right to initalize {' then '.join(self.objs)}. Press ESC when done.")
            rois = []
            while len(rois) < self.n:
                rois = cv.selectROIs(
                    f"image{i}", image, fromCenter=False, showCrosshair=True
                )
                if len(rois) != self.n:
                    print(f"Please select {self.n} objects.")
                    rois = []
            for roi in rois:
                tracker = TRACKERS[self.type]()
                trackers.add(tracker, image, tuple(roi))
        self.initialized = True


    def get_state(self, images):
        if not self.initialized:
            if not self.initialize_trackers(images):
                return None, None

        state = np.full((self.m, self.n, 3), -1)

        for i, (image, trackers) in enumerate(zip(images, self.trackerss)):
            (success, boxes) = trackers.update(image)
            if success:
                points = []
                for j, box in enumerate(boxes):
                    color = COLORS[j % len(COLORS)]
                    x, y, w, h = box
                    state[i, j] = (x + w / 2, y + h / 2, 1)
                    x, y, w, h = map(int, (x, y, w, h))
                    cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    point = state[i, j, :2].astype(int)
                    points.append(point)
                    cv.circle(image, tuple(point), 2, color, 2)
                color = COLORS[len(boxes) % len(COLORS)]
                for j in range(len(points) - 1):
                    cv.line( image, tuple(points[j]), tuple(points[j + 1]), color, 2)
                    distance = la.norm(points[j] - points[j + 1])
                    centroid = np.mean([points[j], points[j + 1]], axis=0).astype(int)
                    cv.putText(
                        image,
                        f"{distance:.2f}",
                        tuple(centroid),
                        cv.FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        2
                    )
            cv.imshow(f"image{i}", image)

        key = cv.waitKey(1)
        if (key == ord('r')):
            self.initialized = False
            state = np.full((self.m, self.n, 3), -1)
        return state, images
