from typing import List
from perception import Perception
import cv2 as cv
import numpy as np

TRACKERS = {
    "csrt": cv.TrackerCSRT_create,
    "kcf": cv.TrackerKCF_create,
    "boosting": cv.TrackerBoosting_create,
    "mil": cv.TrackerMIL_create,
    "tld": cv.TrackerTLD_create,
    "medianflow": cv.TrackerMedianFlow_create,
    "mosse": cv.TrackerMOSSE_create
}

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
            for roi in rois:
                tracker = TRACKERS[self.type]()
                trackers.add(tracker, image, tuple(roi))
        self.initialized = True
            

    def get_state(self, images):
        if not self.initialized:
            self.initialize_trackers(images)

        state = np.ones((self.m, self.n, 3))

        for i, (image, trackers) in enumerate(zip(images, self.trackerss)):
            (success, boxes) = trackers.update(image)
            if success:
                for j, box in enumerate(boxes):
                    x, y, w, h = box
                    state[i, j, :2] = (x + w / 2, y + h / 2)
                    x, y, w, h = map(int, (x, y, w, h))
                    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.imshow(f"image{i}", image)
        key = cv.waitKey(1)
        if (key == ord('r')):
            self.initialized = False
        return state

        

