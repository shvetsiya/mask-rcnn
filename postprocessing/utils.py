import numpy as np
from scipy.ndimage.morphology import binary_fill_holes


def post_process(original_multi_mask: np.array):
    multi_mask = np.copy(original_multi_mask)
    multi_mask = fill_holes(multi_mask)
    multi_mask = filter_small(multi_mask, 8)
    return multi_mask


def fill_holes(multi_mask: np.array):
    for color in range(1, multi_mask.max() + 1):
        bitmask = (multi_mask == color)
        no_holes_bitmask = binary_fill_holes(bitmask)
        multi_mask[no_holes_bitmask] = color
    return multi_mask


def filter_small(multi_mask, threshold):
    color = 0
    for i in range(multi_mask.max()):
        thresh = (multi_mask == (i + 1))

        area = thresh.sum()
        if area < threshold:
            multi_mask[thresh] = 0
        else:
            multi_mask[thresh] = (color + 1)
            color = color + 1

    return multi_mask
