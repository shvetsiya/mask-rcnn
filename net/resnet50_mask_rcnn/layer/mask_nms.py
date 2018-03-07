from common import *
from net.lib.box.process import *
from utility.draw import *


def make_empty_masks(cfg, mode, inputs):  #<todo>
    masks = []
    batch_size, C, H, W = inputs.size()
    for b in range(batch_size):
        mask = np.zeros((H, W), np.float32)
        masks.append(mask)
    return masks


# def mask_nms( cfg, mode, inputs, proposals, mask_logits):
#
#     score_threshold = cfg.mask_test_score_threshold
#     mask_threshold  = cfg.mask_test_mask_threshold
#
#     proposals   = proposals.cpu().data.numpy()
#     mask_logits = mask_logits.cpu().data.numpy()
#     mask_probs  = np_sigmoid(mask_logits)
#
#     masks = []
#     batch_size,C,H,W = inputs.size()
#     for b in range(batch_size):
#         mask  = np.zeros((H,W),np.float32)
#         index = np.where(proposals[:,0]==b)[0]
#
#         instance_id=1
#         if len(index) != 0:
#             for i in index:
#                 p = proposals[i]
#                 prob = p[5]
#                 #print(prob)
#                 if prob>score_threshold:
#                     x0,y0,x1,y1 = p[1:5].astype(np.int32)
#                     h, w = y1-y0+1, x1-x0+1
#                     label = int(p[6]) #<todo>
#                     crop = mask_probs[i, label]
#                     crop = cv2.resize(crop, (w,h), interpolation=cv2.INTER_LINEAR)
#                     crop = crop>mask_threshold
#
#                     mask[y0:y1+1,x0:x1+1] = crop*instance_id + (1-crop)*mask[y0:y1+1,x0:x1+1]
#                     instance_id = instance_id+1
#
#                 if 0: #<debug>
#
#                     images = inputs.data.cpu().numpy()
#                     image = (images[b].transpose((1,2,0))*255).astype(np.uint8)
#                     image = np.clip(image.astype(np.float32)*4,0,255)
#
#                     image_show('image',image,2)
#                     image_show('mask',mask/mask.max()*255,2)
#                     cv2.waitKey(1)
#
#             #<todo>
#             #non-max-suppression to remove overlapping segmentation
#
#         masks.append(mask)
#     return masks


def mask_nms(cfg, mode, inputs, proposals, mask_logits):
    #images = (inputs.data.cpu().numpy().transpose((0,2,3,1))*255).astype(np.uint8)

    overlap_threshold = cfg.mask_test_nms_overlap_threshold
    pre_score_threshold = cfg.mask_test_nms_pre_score_threshold
    mask_threshold = cfg.mask_test_mask_threshold

    proposals = proposals.cpu().data.numpy()
    mask_logits = mask_logits.cpu().data.numpy()
    mask_probs = np_sigmoid(mask_logits)

    masks = []
    batch_size, C, H, W = inputs.size()
    for b in range(batch_size):
        mask = np.zeros((H, W), np.float32)
        index = np.where((proposals[:, 0] == b) & (proposals[:, 5] > pre_score_threshold))[0]

        if len(index) != 0:

            instance = []
            box = []
            for i in index:
                m = np.zeros((H, W), np.bool)

                x0, y0, x1, y1 = proposals[i, 1:5].astype(np.int32)
                h, w = y1 - y0 + 1, x1 - x0 + 1
                label = int(proposals[i, 6])
                crop = mask_probs[i, label]
                crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
                crop = crop > mask_threshold
                m[y0:y1 + 1, x0:x1 + 1] = crop

                instance.append(m)
                box.append((x0, y0, x1, y1))

                #<debug>----------------------------------------------
                if 0:

                    images = inputs.data.cpu().numpy()
                    image = (images[b].transpose((1, 2, 0)) * 255).astype(np.uint8)
                    image = np.clip(image.astype(np.float32) * 4, 0, 255)

                    image_show('image', image, 2)
                    image_show('mask', mask / mask.max() * 255, 2)
                    cv2.waitKey(1)

                #<debug>----------------------------------------------
            instance = np.array(instance, np.bool)
            box = np.array(box, np.float32)

            #compute overlap
            box_overlap = cython_box_overlap(box, box)

            L = len(index)
            instance_overlap = np.zeros((L, L), np.float32)
            for i in range(L):
                instance_overlap[i, i] = 1
                for j in range(i + 1, L):
                    if box_overlap[i, j] < 0.01: continue

                    x0 = int(min(box[i, 0], box[j, 0]))
                    y0 = int(min(box[i, 1], box[j, 1]))
                    x1 = int(max(box[i, 2], box[j, 2]))
                    y1 = int(max(box[i, 3], box[j, 3]))

                    intersection = (instance[i, y0:y1, x0:x1] & instance[j, y0:y1, x0:x1]).sum()
                    area = (instance[i, y0:y1, x0:x1] | instance[j, y0:y1, x0:x1]).sum()
                    instance_overlap[i, j] = intersection / (area + 1e-12)
                    instance_overlap[j, i] = instance_overlap[i, j]

            #non-max suppress
            score = proposals[index, 5]
            index = list(np.argsort(-score))

            ##  https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
            keep = []
            while len(index) > 0:
                i = index[0]
                keep.append(i)
                delete_index = list(np.where(instance_overlap[i] > overlap_threshold)[0])
                index = [e for e in index if e not in delete_index]

                #<todo> : merge?

            for i, k in enumerate(keep):
                mask[np.where(instance[k])] = i + 1

        masks.append(mask)
    return masks


##-----------------------------------------------------------------------------
#if __name__ == '__main__':
#    print( '%s: calling main function ... ' % os.path.basename(__file__))
#
#
#
#
