from common import *
import itertools

from net.lib.box.process import *


#------------------------------------------------------------------------------
# make windows
def make_bases(base_size, base_apsect_ratios):
    bases = []
    for ratio in base_apsect_ratios:
        w = ratio[0] * base_size
        h = ratio[1] * base_size
        rw = round(w / 2)
        rh = round(h / 2)
        base = (
            -rw,
            -rh,
            rw,
            rh,
        )
        bases.append(base)

    bases = np.array(bases, np.float32)
    return bases


def make_windows(f, scale, bases):
    windows = []
    _, _, H, W = f.size()
    for y, x in itertools.product(range(H), range(W)):
        cx = x * scale
        cy = y * scale
        for b in bases:
            x0, y0, x1, y1 = b
            x0 += cx
            y0 += cy
            x1 += cx
            y1 += cy
            windows.append([x0, y0, x1, y1])

    windows = np.array(windows, np.float32)
    return windows


def make_rpn_windows(cfg, fs):

    rpn_windows = []
    num_scales = len(cfg.rpn_scales)
    for l in range(num_scales):
        bases = make_bases(cfg.rpn_base_sizes[l], cfg.rpn_base_apsect_ratios[l])
        windows = make_windows(fs[l], cfg.rpn_scales[l], bases)
        rpn_windows.append(windows)

    rpn_windows = np.vstack(rpn_windows)

    return rpn_windows


# "UnitBox: An Advanced Object Detection Network" - Jiahui Yu, Yuning Jiang, Zhangyang Wang, Zhimin Cao, Thomas Huang
#  https://arxiv.org/abs/1608.01471


def rpn_encode(window, truth_box):
    cx = (window[:, 0] + window[:, 2]) / 2
    cy = (window[:, 1] + window[:, 3]) / 2
    w = (window[:, 2] - window[:, 0] + 1)
    h = (window[:, 3] - window[:, 1] + 1)

    target = (truth_box - np.column_stack([cx, cy, cx, cy])) / np.column_stack([w, h, w, h])
    target = target * np.array([-1, -1, 1, 1], np.float32)
    return target


def rpn_decode(window, delta):
    cx = (window[:, 0] + window[:, 2]) / 2
    cy = (window[:, 1] + window[:, 3]) / 2
    w = (window[:, 2] - window[:, 0] + 1)
    h = (window[:, 3] - window[:, 1] + 1)

    delta = delta * np.array([-1, -1, 1, 1], np.float32)
    box = delta * np.column_stack([w, h, w, h]) + np.column_stack([cx, cy, cx, cy])

    return box


## faster-rcnn box encode/decode  ---------------------------------
# def rpn_encode(window, truth_box):
#     return box_transform(window, truth_box)
#
# def rpn_decode(window, delta):
#     return  box_transform_inv(window, delta)
#
#

# this is in gpu ##################################################


def rpn_nms(cfg, mode, inputs, window, logits_flat, deltas_flat):

    if mode in [
            'train',
    ]:
        nms_pre_score_threshold = cfg.rpn_train_nms_pre_score_threshold
        nms_overlap_threshold = cfg.rpn_train_nms_overlap_threshold
        nms_min_size = cfg.rpn_train_nms_min_size

    elif mode in [
            'eval',
            'valid',
            'test',
    ]:
        nms_pre_score_threshold = cfg.rpn_test_nms_pre_score_threshold
        nms_overlap_threshold = cfg.rpn_test_nms_overlap_threshold
        nms_min_size = cfg.rpn_test_nms_min_size

        if mode in ['eval']:
            nms_pre_score_threshold = 0.05  # set low numbe r to make roc curve.

    else:
        raise ValueError('rpn_nms(): invalid mode = %s?' % mode)

    logits = logits_flat.data.cpu().numpy()
    deltas = deltas_flat.data.cpu().numpy()
    batch_size, _, height, width = inputs.size()
    num_classes = cfg.num_classes

    proposals = []
    for b in range(batch_size):
        proposal = [
            np.empty((0, 7), np.float32),
        ]

        ps = np_softmax(logits[b])
        ds = deltas[b]

        for c in range(1, num_classes):  #skip background  #num_classes
            index = np.where(ps[:, c] > nms_pre_score_threshold)[0]
            if len(index) > 0:
                p = ps[index, c].reshape(-1, 1)
                d = ds[index, c]
                w = window[index]
                box = rpn_decode(w, d)
                box = clip_boxes(box, width, height)

                keep = filter_boxes(box, min_size=nms_min_size)
                if len(keep) > 0:
                    box = box[keep]
                    p = p[keep]
                    keep = gpu_nms(np.hstack((box, p)), nms_overlap_threshold)

                    prop = np.zeros((len(keep), 7), np.float32)
                    prop[:, 0] = b
                    prop[:, 1:5] = np.around(box[keep], 0)
                    prop[:, 5] = p[keep, 0]
                    prop[:, 6] = c
                    proposal.append(prop)

        proposal = np.vstack(proposal)
        proposals.append(proposal)

    proposals = Variable(torch.from_numpy(np.vstack(proposals))).cuda()
    return proposals


#-----------------------------------------------------------------------------
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
