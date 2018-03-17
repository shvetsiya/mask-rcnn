from typing import Tuple

from common import *
from net.lib.box.process import *
from utility.draw import *


class BoundingBox(object):
    """A class which represents bounding box.
    """

    def __init__(self, coordinates: Tuple[int, int, int, int], score: float, color: int):
        self.coordinates = coordinates
        self.score = score
        self.color = color
        assert color != 0, 'Bounding box should not be related to the background'


def masks_nms_for_batch(cfg, inputs, proposals, masks_logit):
    """Given inputs, proposals, and masks logits for a batch, returns a list of multi masks and
    corresponding bounding boxes.

    Args:
        cfg: Configuration setup.
        inputs: Net's inputs.
        proposals: Numpy array of size 7:
            proposal[k, 0]: index in batch (index of corresponding input in the batch);
            proposal[k, 1:5]: (x0, y0, x1, y1) -- coordinates of the bounding box;
            proposal[k, 5]: score of the bounding box;
            proposal[k, 6]: label, whatever it means.
        masks_logit: Not-normalized masks (same length as proposals).

    Returns:
        masks_for_batch: Array of multi-masks with same length as inputs (batch size).
        bounding_boxes_for_batch: Array of bounding boxes for each multi_mask. Each bounding box
            corresponds to one of the masks_for_batch and stores score and color in the mask.
    """
    overlap_threshold = cfg.mask_test_nms_overlap_threshold
    pre_score_threshold = cfg.mask_test_nms_pre_score_threshold
    mask_threshold = cfg.mask_test_mask_threshold

    proposals = proposals.cpu().data.numpy()
    masks_logit = masks_logit.cpu().data.numpy()
    masks_prob = np_sigmoid(masks_logit)

    masks_for_batch = []
    bounding_boxes_for_batch = []
    batch_size, _, image_h, image_w = inputs.size()

    for index_in_batch in range(batch_size):
        boxes_indices = np.where((proposals[:, 0] == index_in_batch) &
                                 (proposals[:, 5] > pre_score_threshold))[0]

        if len(boxes_indices) == 0:
            mask = np.zeros((image_h, image_w), np.float32)
            masks_for_batch.append(mask)
            bounding_boxes_for_batch.append([])
            continue

        instances = []
        boxes = []
        for box_index in boxes_indices:
            x0, y0, x1, y1 = proposals[box_index, 1:5].astype(np.int32)
            box_h, box_w = y1 - y0 + 1, x1 - x0 + 1
            label = int(proposals[box_index, 6])
            crop = masks_prob[box_index, label]  # 28x28
            crop = cv2.resize(crop, (box_w, box_h), interpolation=cv2.INTER_LINEAR)
            crop = crop > mask_threshold

            instance_mask = np.zeros((image_h, image_w), np.bool)
            instance_mask[y0:y1 + 1, x0:x1 + 1] = crop

            instances.append(instance_mask)
            boxes.append((x0, y0, x1, y1))

        instances = np.array(instances, np.bool)
        boxes = np.array(boxes, np.float32)

        # Compute overlap
        boxes_overlap = cython_box_overlap(boxes, boxes)

        boxes_num = len(boxes_indices)
        instance_overlap = np.zeros((boxes_num, boxes_num), np.float32)
        for first_index in range(boxes_num):
            instance_overlap[first_index, first_index] = 1
            for second_index in range(first_index + 1, boxes_num):
                if boxes_overlap[first_index, second_index] < 0.01: continue

                x0, y0 = np.minimum(boxes[first_index, 0:2], boxes[second_index, 0:2]).astype(int)
                x1, y1 = np.maximum(boxes[first_index, 2:], boxes[second_index, 2:]).astype(int)

                intersection = (instances[first_index, y0:y1, x0:x1] &
                                instances[second_index, y0:y1, x0:x1]).sum()
                union = (instances[first_index, y0:y1, x0:x1] |
                         instances[second_index, y0:y1, x0:x1]).sum()
                intersection_over_union = intersection / (union + 1e-12)
                instance_overlap[first_index, second_index] = intersection_over_union
                instance_overlap[second_index, first_index] = intersection_over_union

        # Non-max suppress
        boxes_score = proposals[boxes_indices, 5]
        boxes_indices = list(np.argsort(-boxes_score))

        # https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
        keep = []
        while len(boxes_indices) > 0:
            i = boxes_indices[0]  # with current maximum score
            keep.append(i)
            delete_index = list(np.where(instance_overlap[i] > overlap_threshold)[0])
            boxes_indices = [e for e in boxes_indices if e not in delete_index]

        mask = np.zeros((image_h, image_w), np.uint32)
        bounding_boxes = []
        for color, box_index in enumerate(keep, 1):
            mask[np.where(instances[box_index])] = color
            bounding_boxes.append(BoundingBox(boxes[box_index], boxes_score[box_index], color))

        masks_for_batch.append(mask)
        bounding_boxes_for_batch.append(bounding_boxes)

    assert len(masks_for_batch) == len(bounding_boxes_for_batch), 'Should be same size'

    return masks_for_batch, bounding_boxes_for_batch
