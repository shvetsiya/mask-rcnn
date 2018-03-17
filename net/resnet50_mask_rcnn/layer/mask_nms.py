from common import *
from net.lib.box.process import *
from utility.draw import *


def make_empty_masks(inputs):
    batch_size, _, image_h, image_w = inputs.size()
    masks = []
    for index_in_batch in range(batch_size):
        mask = np.zeros((image_h, image_w), np.float32)
        masks.append(mask)
    return masks


def mask_nms(cfg, mode, inputs, proposals, mask_logits):
    overlap_threshold = cfg.mask_test_nms_overlap_threshold
    pre_score_threshold = cfg.mask_test_nms_pre_score_threshold
    mask_threshold = cfg.mask_test_mask_threshold

    proposals = proposals.cpu().data.numpy()
    mask_logits = mask_logits.cpu().data.numpy()
    mask_probs = np_sigmoid(mask_logits)

    masks = []
    batch_size, _, image_h, image_w = inputs.size()

    for index_in_batch in range(batch_size):
        boxes_indices = np.where((proposals[:, 0] == index_in_batch) &
                                 (proposals[:, 5] > pre_score_threshold))[0]

        if len(boxes_indices) == 0:
            mask = np.zeros((image_h, image_w), np.float32)
            masks.append(mask)
            continue

        instances = []
        boxes = []
        for box_index in boxes_indices:
            x0, y0, x1, y1 = proposals[box_index, 1:5].astype(np.int32)
            box_h, box_w = y1 - y0 + 1, x1 - x0 + 1
            label = int(proposals[box_index, 6])
            crop = mask_probs[box_index, label]  # 28x28
            crop = cv2.resize(crop, (box_w, box_h), interpolation=cv2.INTER_LINEAR)
            crop = crop > mask_threshold

            instance_mask = np.zeros((image_h, image_w), np.bool)
            instance_mask[y0:y1 + 1, x0:x1 + 1] = crop

            instances.append(instance_mask)
            boxes.append((x0, y0, x1, y1))

        instances = np.array(instances, np.bool)
        boxes = np.array(boxes, np.float32)

        #compute overlap
        box_overlap = cython_box_overlap(boxes, boxes)

        boxes_num = len(boxes_indices)
        instance_overlap = np.zeros((boxes_num, boxes_num), np.float32)
        for first_index in range(boxes_num):
            instance_overlap[first_index, first_index] = 1
            for second_index in range(first_index + 1, boxes_num):
                if box_overlap[first_index, second_index] < 0.01: continue

                x0, y0 = int(min(boxes[first_index, 0:2], boxes[second_index, 0:2]))
                x1, y1 = int(max(boxes[first_index, 2:], boxes[second_index, 2:]))

                intersection = (instances[first_index, y0:y1, x0:x1] &
                                instances[second_index, y0:y1, x0:x1]).sum()
                union = (instances[first_index, y0:y1, x0:x1] |
                         instances[second_index, y0:y1, x0:x1]).sum()
                intersection_over_union = intersection / (union + 1e-12)
                instance_overlap[first_index, second_index] = intersection_over_union
                instance_overlap[second_index, first_index] = intersection_over_union

        #non-max suppress
        box_scores = proposals[boxes_indices, 5]
        boxes_indices = list(np.argsort(-box_scores))

        ##  https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
        keep = []
        while len(boxes_indices) > 0:
            i = boxes_indices[0]  # with current maximum score
            keep.append(i)
            delete_index = list(np.where(instance_overlap[i] > overlap_threshold)[0])
            boxes_indices = [e for e in boxes_indices if e not in delete_index]

        mask = np.zeros((image_h, image_w), np.float32)
        for i, k in enumerate(keep):
            mask[np.where(instances[k])] = i + 1

        masks.append(mask)

    return masks
