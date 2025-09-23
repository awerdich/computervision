""" Methods from the supervision library """

import numpy as np

def xcycwh_to_xyxy(xcycwh: np.ndarray) -> np.ndarray:
    """
    Converts bounding box coordinates from `(center_x, center_y, width, height)`
    format to `(x_min, y_min, x_max, y_max)` format.
    """
    xyxy = xcycwh.copy()
    xyxy[:, 0] = xcycwh[:, 0] - xcycwh[:, 2] / 2
    xyxy[:, 1] = xcycwh[:, 1] - xcycwh[:, 3] / 2
    xyxy[:, 2] = xcycwh[:, 0] + xcycwh[:, 2] / 2
    xyxy[:, 3] = xcycwh[:, 1] + xcycwh[:, 3] / 2
    return xyxy

