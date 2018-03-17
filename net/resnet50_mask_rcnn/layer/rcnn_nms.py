from common import *
from utility.draw import *

from net.lib.box.process import *


## faster-rcnn box encode/decode  ---------------------------------
def rcnn_encode(window, truth_box):
    return box_transform(window, truth_box)


def rcnn_decode(window, delta):
    return box_transform_inv(window, delta)


# def draw_rcnn_pre_nms(image, probs, deltas, proposals, cfg, colors, names, threshold=-1, is_before=1, is_after=1):
#
#     height,width = image.shape[0:2]
#     num_classes  = cfg.num_classes
#
#     probs  = probs.cpu().data.numpy()
#     deltas = deltas.cpu().data.numpy()
#     proposals    = proposals.data.cpu().numpy()
#     num_proposals = len(proposals)
#
#     labels = np.argmax(probs,axis=1)
#     probs  = probs[range(0,num_proposals),labels]
#     idx    = np.argsort(probs)
#     for j in range(num_proposals):
#         i = idx[j]
#
#         s = probs[i]
#         l = labels[i]
#         if s<threshold or l==0:
#             continue
#
#         a = proposals[i, 0:4]
#         t = deltas[i,l*4:(l+1)*4]
#         b = box_transform_inv(a.reshape(1,4), t.reshape(1,4))
#         b = clip_boxes(b, width, height)  ## clip here if you have drawing error
#         b = b.reshape(-1)
#
#         #a   = a.astype(np.int32)
#         color = (s*np.array(colors[l])).astype(np.uint8)
#         color = (int(color[0]),int(color[1]),int(color[2]))
#         if is_before==1:
#             draw_dotted_rect(image,(a[0], a[1]), (a[2], a[3]), color, 1)
#             #cv2.rectangle(image,(a[0], a[1]), (a[2], a[3]), (int(color[0]),int(color[1]),int(color[2])), 1)
#
#         if is_after==1:
#             cv2.rectangle(image,(b[0], b[1]), (b[2], b[3]), color, 1)
#
#         draw_shadow_text(image , '%f'%s,(b[0], b[1]), 0.5, (255,255,255), 1, cv2.LINE_AA)
#
#
#
# #---------------------------------------------------------------------------
#
#this is in cpu: <todo> change to gpu ?
def rcnn_nms(cfg, mode, inputs, proposals, logits, deltas):
    if mode in ['train']:
        nms_pre_score_threshold = cfg.rcnn_train_nms_pre_score_threshold
        nms_overlap_threshold = cfg.rcnn_train_nms_overlap_threshold
        nms_min_size = cfg.rcnn_train_nms_min_size
    elif mode in ['valid', 'test', 'eval']:
        nms_pre_score_threshold = cfg.rcnn_test_nms_pre_score_threshold
        nms_overlap_threshold = cfg.rcnn_test_nms_overlap_threshold
        nms_min_size = cfg.rcnn_test_nms_min_size

        if mode in ['eval']:
            nms_pre_score_threshold = 0.05  # set low numbe r to make roc curve.
    else:
        raise ValueError('rcnn_nms(): invalid mode = %s?' % mode)

    batch_size, _, height, width = inputs.size()  #original image width
    num_classes = cfg.num_classes

    probs = np_sigmoid(logits.cpu().data.numpy())
    deltas = deltas.cpu().data.numpy().reshape(-1, num_classes, 4)
    proposals = proposals.cpu().data.numpy()

    #non-max suppression
    detections = []
    for b in range(batch_size):
        detection = [
            np.empty((0, 7), np.float32),
        ]

        index = np.where(proposals[:, 0] == b)[0]
        if len(index) > 0:
            prob = probs[index]
            delta = deltas[index]
            proposal = proposals[index]

            for j in range(1, num_classes):  #skip background
                idx = np.where(prob[:, j] > nms_pre_score_threshold)[0]
                if len(idx) > 0:
                    p = prob[idx, j].reshape(-1, 1)
                    d = delta[idx, j]
                    box = rcnn_decode(proposal[idx, 1:5], d)
                    box = clip_boxes(box, width, height)

                    keep = filter_boxes(box, min_size=nms_min_size)
                    num = len(keep)
                    if num > 0:
                        box = box[keep]
                        p = p[keep]
                        keep = gpu_nms(np.hstack((box, p)), nms_overlap_threshold)

                        det = np.zeros((num, 7), np.float32)
                        det[:, 0] = b
                        det[:, 1:5] = np.around(box, 0)
                        det[:, 5] = p[:, 0]
                        det[:, 6] = j
                        detection.append(det)

        detection = np.vstack(detection)

        ##limit to MAX_PER_IMAGE detections over all classes
        # if nms_max_per_image > 0:
        #     if len(detection) > nms_max_per_image:
        #         threshold = np.sort(detection[:,4])[-nms_max_per_image]
        #         keep = np.where(detection[:,4] >= threshold)[0]
        #         detection = detection[keep, :]

        detections.append(detection)

    detections = Variable(torch.from_numpy(np.vstack(detections))).cuda()
    return detections


#-----------------------------------------------------------------------------
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
