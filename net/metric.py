#from common import *

import numpy as np
import pandas as pd
import cv2
from utility.draw import *
from net.lib.box.overlap.cython_box_overlap import cython_box_overlap
#from net.lib.box.process import *


def run_length_encode(x):
    bs = np.where(x.T.flatten())[0]

    rle = []
    prev = -2
    for b in bs:
        if (b > prev + 1): rle.extend((b + 1, 0))
        rle[-1] += 1
        prev = b

    #https://www.kaggle.com/c/data-science-bowl-2018/discussion/48561#
    if len(rle) != 0 and rle[-1] + rle[-2] == x.size:
        rle[-2] = rle[-2] - 1  #print('xxx')

    rle = ' '.join([str(r) for r in rle])
    return rle


#https://www.kaggle.com/stainsby/fast-tested-rle-and-input-routines
# def run_length_encode(x):
#     pixels = x.T.flatten()
#     # We need to allow for cases where there is a '1' at either end of the sequence.
#     # We do this by padding with a zero at each end when needed.
#     use_padding = False
#     if pixels[0] or pixels[-1]:
#         use_padding = True
#         pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
#         pixel_padded[1:-1] = pixels
#         pixels = pixel_padded
#     rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
#     if use_padding:
#         rle = rle - 1
#     rle[1::2] = rle[1::2] - rle[:-1:2]
#
#     #https://www.kaggle.com/c/data-science-bowl-2018/discussion/48561#
#     if len(rle)!=0 and rle[-1]+rle[-2] == len(pixels):
#         rle[-2] = rle[-2] -1  #print('xxx')
#
#     rle = ' '.join([str(r) for r in rle])
#     return rle


def run_length_decode(rle, H, W, fill_value=255):

    mask = np.zeros((H * W), np.uint8)
    rle = np.array([int(s) for s in rle.split(' ')]).reshape(-1, 2)
    for r in rle:
        start = r[0] - 1
        end = start + r[1]
        mask[start:end] = fill_value
    mask = mask.reshape(W, H).T  # H, W need to swap as transposing.
    return mask


#https://www.kaggle.com/wcukierski/example-metric-implementation
def compute_precision(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn


def print_precision(precision):

    print('thresh   prec    TP    FP    FN')
    print('---------------------------------')
    for (t, p, tp, fp, fn) in precision:
        print('%0.2f     %0.2f   %3d   %3d   %3d' % (t, p, tp, fp, fn))


def compute_average_precision_for_mask(predict, truth, t_range=np.arange(0.5, 1.0, 0.05)):

    num_truth = len(np.unique(truth))
    num_predict = len(np.unique(predict))

    # Compute intersection between all objects
    intersection = np.histogram2d(
        truth.flatten(), predict.flatten(), bins=(num_truth, num_predict))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(truth, bins=num_truth)[0]
    area_pred = np.histogram(predict, bins=num_predict)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    precision = []
    average_precision = 0
    for t in t_range:
        tp, fp, fn = compute_precision(t, iou)
        p = tp / (tp + fp + fn)
        precision.append((t, p, tp, fp, fn))
        average_precision += p

    average_precision /= len(precision)
    return average_precision, precision


#one class only-----------------------------------------------------------------
HIT = 1
MISS = 0
TP = 1
FP = 0
INVALID = -1


def compute_precision_for_box(box, truth_box, truth_label, threshold=[0.5]):
    num_truth_box = len(truth_box)
    num_box = len(box)

    overlap = cython_box_overlap(box, truth_box)
    argmax_overlap = np.argmax(overlap, 0)
    max_overlap = overlap[argmax_overlap, np.arange(num_truth_box)]

    invalid_truth_box = truth_box[truth_label < 0]
    invalid_valid_overlap = cython_box_overlap(box, invalid_truth_box)

    precision = []
    recall = []
    result = []
    truth_result = []

    for t in threshold:
        truth_r = np.ones(num_truth_box, np.int32)
        r = np.ones(num_box, np.int32)

        # truth_result
        truth_r[...] = INVALID
        truth_r[(max_overlap < t) & (truth_label > 0)] = MISS
        truth_r[(max_overlap >= t) & (truth_label > 0)] = HIT

        # result
        r[...] = FP
        r[argmax_overlap[truth_r == HIT]] = TP

        index = np.where(r == FP)[0]
        if len(index) > 0:
            index = index[np.where(invalid_valid_overlap[index] > t)[0]]
            r[index] = INVALID

        ##-----------------------------
        num_truth = (truth_r != INVALID).sum()
        num_hit = (truth_r == HIT).sum()
        num_miss = (truth_r == MISS).sum()
        rec = num_hit / num_truth

        num_tp = (r == TP).sum()
        num_fp = (r == FP).sum()
        prec = num_tp / max(num_tp + num_fp + num_miss, 1e-12)

        precision.append(prec)
        recall.append(rec)
        result.append(r)
        truth_result.append(truth_r)

        # if len(thresholds)==1:
        #     precisions = precisions[0]
        #     recalls = recalls[0]
        #     results = results[0]
        #     truth_results = truth_results[0]

    return precision, recall, result, truth_result


def compute_hit_fp_for_box(proposals, truth_boxes, truth_labels):

    score = []
    hit = []
    fp = []
    num_miss = 0

    for (proposal, truth_box, truth_label) in zip(proposals, truth_boxes, truth_labels):

        box = proposal[:, 1:5]
        precision, recall, result, truth_result = compute_precision_for_box(
            box, truth_box, truth_label, threshold=[0.5])
        result, truth_result = result[0], truth_result[0]

        s = proposal[:, 5]
        N = len(result)
        h = np.zeros(N)
        f = np.zeros(N)
        h[np.where(result == HIT)] = 1
        f[np.where(result == FP)] = 1

        num_miss = (truth_result == MISS).sum()
        hit = hit + list(h)
        fp = fp + list(f)
        score = score + list(s)

    return hit, fp, score, num_miss


# check #################################################################
def run_check_run_length_encode():

    name = 'b98681c74842c4058bd2f88b06063731c26a90da083b1ef348e0ec734c58752b'

    npy_file = DATA_DIR + '/image/stage1_train/' + name + '/multi_mask.npy'
    multi_mask = np.load(npy_file)

    cvs_EncodedPixels = []
    num = int(multi_mask.max())
    for m in range(num):
        rle = run_length_encode(multi_mask == m + 1)
        cvs_EncodedPixels.append(rle)
    cvs_EncodedPixels.sort()

    #reference encoding from 'stage1_train_labels.csv'
    df = pd.read_csv(DATA_DIR + '/__download__/stage1_train_labels.csv')
    df = df.loc[df['ImageId'] == name]

    reference_cvs_EncodedPixels = df['EncodedPixels'].values
    reference_cvs_EncodedPixels.sort()

    print('reference_cvs_EncodedPixels\n', reference_cvs_EncodedPixels)
    print('')
    print('cvs_EncodedPixels\n', cvs_EncodedPixels)
    print('')

    print(reference_cvs_EncodedPixels == cvs_EncodedPixels)


def run_check_run_length_decode():

    name = 'b98681c74842c4058bd2f88b06063731c26a90da083b1ef348e0ec734c58752b'

    npy_file = DATA_DIR + '/image/stage1_train/' + name + '/multi_mask.npy'
    multi_mask = np.load(npy_file)
    H, W = multi_mask.shape[:2]

    cvs_EncodedPixels = []
    num = int(multi_mask.max())
    for m in range(num):
        rle = run_length_encode(multi_mask == m + 1)
        cvs_EncodedPixels.append(rle)

    #reference encoding from 'stage1_train_labels.csv'
    df = pd.read_csv(DATA_DIR + '/__download__/stage1_train_labels.csv')
    df = df.loc[df['ImageId'] == name]

    reference_cvs_EncodedPixels = df['EncodedPixels'].values
    reference_cvs_EncodedPixels.sort()

    reference_multi_mask = np.zeros((H, W), np.int32)
    for rle in reference_cvs_EncodedPixels:
        thresh = run_length_decode(rle, H, W, fill_value=255)
        id = cvs_EncodedPixels.index(rle)
        reference_multi_mask[thresh > 128] = id + 1

    reference_multi_mask = reference_multi_mask.astype(np.float32)
    reference_multi_mask = reference_multi_mask / reference_multi_mask.max() * 255
    multi_mask = multi_mask.astype(np.float32)
    multi_mask = multi_mask / multi_mask.max() * 255

    print((reference_multi_mask != multi_mask).sum())

    image_show('multi_mask', multi_mask, 2)
    image_show('reference_multi_mask', reference_multi_mask, 2)
    image_show('diff', (reference_multi_mask != multi_mask) * 255, 2)
    cv2.waitKey(0)


def run_check_compute_precision_for_box():

    H, W = 256, 256

    truth_label = np.array([1, 1, 2, 1, -1], np.float32)
    truth_box = np.array([
        [
            10,
            10,
            0,
            0,
        ],
        [
            100,
            10,
            0,
            0,
        ],
        [
            50,
            50,
            0,
            0,
        ],
        [
            10,
            100,
            0,
            0,
        ],
        [
            100,
            100,
            0,
            0,
        ],
    ], np.float32)
    truth_box[:, 2] = truth_box[:, 0] + 25
    truth_box[:, 3] = truth_box[:, 1] + 25

    box = np.zeros((7, 4), np.float32)
    box[:5] = truth_box[[0, 1, 2, 4, 3]] + np.random.uniform(-10, 10, size=(5, 4))
    box[5] = [
        10,
        10,
        80,
        80,
    ]
    box[6] = [
        100,
        100,
        180,
        180,
    ]

    thresholds = [0.3, 0.5, 0.6]
    precisions, recalls, results, truth_results = \
        compute_precision_for_box(box, truth_box, truth_label, thresholds)

    for precision, recall, result, truth_result, threshold in zip(precisions, recalls, results,
                                                                  truth_results, thresholds):
        print('')
        print('threshold ', threshold)
        print('precision ', precision)
        print('recall    ', recall)

        image = np.zeros((H, W, 3), np.uint8)
        for i, b in enumerate(truth_box):
            x0, y0, x1, y1 = b.astype(np.int32)
            if truth_result[i] == HIT:
                draw_screen_rect(image, (x0, y0), (x1, y1), (0, 255, 255), 0.5)
            if truth_result[i] == MISS:
                draw_screen_rect(image, (x0, y0), (x1, y1), (0, 0, 255), 0.5)
            if truth_result[i] == INVALID:
                draw_screen_rect(image, (x0, y0), (x1, y1), (255, 255, 255), 0.5)

        for i, b in enumerate(box):
            x0, y0, x1, y1 = b.astype(np.int32)
            if result[i] == TP:
                cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 255), 1)
            if result[i] == FP:
                cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 1)
            if result[i] == INVALID:
                draw_dotted_rect(image, (x0, y0), (x1, y1), (255, 255, 255), 1)

        image_show("image_box", image, 1)
        cv2.waitKey(0)


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    #run_check_run_length_encode()
    run_check_compute_precision_for_box()

    print('\nsucess!')
