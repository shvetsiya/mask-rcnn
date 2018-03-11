import os, sys
sys.path.append(os.path.dirname(__file__))

import numpy as np

# torch libs
import torch
from torch.autograd import Variable
from torch.nn.parallel.data_parallel import data_parallel
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SequentialSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from common import RESULTS_DIR, IDENTIFIER, SEED, PROJECT_PATH

from utility.file import Logger
from net.resnet50_mask_rcnn.configuration import Configuration
from net.resnet50_mask_rcnn.draw import draw_multi_proposal_metric, draw_mask_metric
from net.resnet50_mask_rcnn.model import MaskRcnnNet
from net.metric import compute_average_precision_for_mask, compute_precision_for_box
from dataset.reader import ScienceDataset, multi_mask_to_annotation, instance_to_multi_mask, \
        multi_mask_to_contour_overlay, multi_mask_to_color_overlay
from dataset.transform import pad_to_factor


## overwrite functions ###
def revert(net, images):

    def torch_clip_proposals(proposals, index, width, height):
        boxes = torch.stack((
            proposals[index, 0],
            proposals[index, 1].clamp(0, width - 1),
            proposals[index, 2].clamp(0, height - 1),
            proposals[index, 3].clamp(0, width - 1),
            proposals[index, 4].clamp(0, height - 1),
            proposals[index, 5],
            proposals[index, 6],
        ), 1)
        return proposals

    # ----

    batch_size = len(images)
    for b in range(batch_size):
        image = images[b]
        height, width = image.shape[:2]

        # net.rpn_logits_flat  <todo>
        # net.rpn_deltas_flat  <todo>
        # net.rpn_window       <todo>
        # net.rpn_proposals    <todo>

        # net.rcnn_logits
        # net.rcnn_deltas
        # net.rcnn_proposals <todo>

        # mask --
        # net.mask_logits
        index = (net.detections[:, 0] == b).nonzero().view(-1)
        net.detections = torch_clip_proposals(net.detections, index, width, height)

        net.masks[b] = net.masks[b][:height, :width]

    return net, image


def eval_augment(image, multi_mask, meta, index):

    pad_image = pad_to_factor(image, factor=16)
    input = torch.from_numpy(pad_image.transpose((2, 0, 1))).float().div(255)
    box, label, instance = multi_mask_to_annotation(multi_mask)

    return input, box, label, instance, meta, image, index


def eval_collate(batch):

    batch_size = len(batch)
    #for b in range(batch_size): print (batch[b][0].size())
    inputs = torch.stack([batch[b][0] for b in range(batch_size)], 0)
    boxes = [batch[b][1] for b in range(batch_size)]
    labels = [batch[b][2] for b in range(batch_size)]
    instances = [batch[b][3] for b in range(batch_size)]
    metas = [batch[b][4] for b in range(batch_size)]
    images = [batch[b][5] for b in range(batch_size)]
    indices = [batch[b][6] for b in range(batch_size)]

    return [inputs, boxes, labels, instances, metas, images, indices]


#--------------------------------------------------------------
def run_evaluate():

    out_dir = RESULTS_DIR + '/mask-rcnn-50-gray500-02'
    initial_checkpoint = \
        RESULTS_DIR + '/mask-rcnn-50-gray500-02/checkpoint/00008500_model.pth'
    ##

    ## setup  ---------------------------
    os.makedirs(out_dir + '/evaluate/overlays', exist_ok=True)
    os.makedirs(out_dir + '/evaluate/npys', exist_ok=True)
    os.makedirs(out_dir + '/checkpoint', exist_ok=True)
    os.makedirs(out_dir + '/backup', exist_ok=True)

    log = Logger()
    log.open(out_dir + '/log.evaluate.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')

    ## net ------------------------------
    cfg = Configuration()
    # cfg.rpn_train_nms_pre_score_threshold = 0.8 #0.885#0.5
    # cfg.rpn_test_nms_pre_score_threshold  = 0.8 #0.885#0.5

    net = MaskRcnnNet(cfg).cuda()
    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(
            torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    log.write('%s\n\n' % (type(net)))
    log.write('\n')

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    test_dataset = ScienceDataset(
        #'train1_ids_gray2_500', mode='train',
        'valid1_ids_gray2_43',
        mode='train',
        #'debug1_ids_gray2_10', mode='train',
        transform=eval_augment)
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=1,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=eval_collate)

    log.write('\ttest_dataset.split = %s\n' % (test_dataset.split))
    log.write('\tlen(test_dataset)  = %d\n' % (len(test_dataset)))
    log.write('\n')

    ## start evaluation here! ##############################################
    log.write('** start evaluation here! **\n')
    mask_average_precisions = []
    box_precisions_50 = []

    test_num = 0
    test_loss = np.zeros(5, np.float32)
    test_acc = 0
    for i, (inputs, truth_boxes, truth_labels, truth_instances, metas, images,
            indices) in enumerate(test_loader, 0):
        if all((truth_label > 0).sum() == 0 for truth_label in truth_labels): continue

        net.set_mode('test')
        with torch.no_grad():
            inputs = Variable(inputs).cuda()
            net(inputs, truth_boxes, truth_labels, truth_instances)
            #loss = net.loss(inputs, truth_boxes,  truth_labels, truth_instances)

        ##save results ---------------------------------------
        revert(net, images)

        batch_size, C, H, W = inputs.size()

        # NOTE: Current version support batch_size==1 for variable size input. To use batch_size>1,
        # need to fix code for net.windows, etc.
        assert (batch_size == 1)

        inputs = inputs.data.cpu().numpy()

        # window          = net.rpn_window
        # rpn_logits_flat = net.rpn_logits_flat.data.cpu().numpy()
        # rpn_deltas_flat = net.rpn_deltas_flat.data.cpu().numpy()
        # proposals  = net.rpn_proposals
        masks = net.masks
        detections = net.detections.cpu().numpy()

        for b in range(batch_size):
            #image0 = (inputs[b].transpose((1,2,0))*255).astype(np.uint8)
            image = images[b]
            height, width = image.shape[:2]
            mask = masks[b]

            index = np.where(detections[:, 0] == b)[0]
            detection = detections[index]
            box = detection[:, 1:5]

            truth_mask = instance_to_multi_mask(truth_instances[b])
            truth_box = truth_boxes[b]
            truth_label = truth_labels[b]
            truth_instance = truth_instances[b]

            mask_average_precision, mask_precision =\
                compute_average_precision_for_mask(mask, truth_mask, t_range=np.arange(0.5, 1.0, 0.05))

            box_precision, box_recall, box_result, truth_box_result = \
                compute_precision_for_box(box, truth_box, truth_label, threshold=[0.5])
            box_precision = box_precision[0]

            mask_average_precisions.append(mask_average_precision)
            box_precisions_50.append(box_precision)

            # --------------------------------------------
            id = test_dataset.ids[indices[b]]
            name = id.split('/')[-1]
            print('%d\t%s\t%0.5f  (%0.5f)' % (i, name, mask_average_precision, box_precision))

            #----
            contour_overlay = multi_mask_to_contour_overlay(mask, image, color=[0, 255, 0])
            color_overlay = multi_mask_to_color_overlay(mask, color='summer')
            color1_overlay = multi_mask_to_contour_overlay(
                mask, color_overlay, color=[255, 255, 255])
            all1 = np.hstack((image, contour_overlay, color1_overlay))

            all6 = draw_multi_proposal_metric(cfg, image, detection, truth_box, truth_label,
                                              [0, 255, 255], [255, 0, 255], [255, 255, 0])
            all7 = draw_mask_metric(cfg, image, mask, truth_box, truth_label, truth_instance)

            #image_show('overlay_mask',overlay_mask)
            #image_show('overlay_truth',overlay_truth)
            #image_show('overlay_error',overlay_error)
            # image_show('all1', all1)
            # image_show('all6', all6)
            # image_show('all7', all7)
            # cv2.waitKey(0)

        # print statistics  ------------
        test_acc += 0  #batch_size*acc[0][0]
        # test_loss += batch_size*np.array((
        #                    loss.cpu().data.numpy(),
        #                    net.rpn_cls_loss.cpu().data.numpy(),
        #                    net.rpn_reg_loss.cpu().data.numpy(),
        #                     0,0,
        #                  ))
        test_num += batch_size

    #assert(test_num == len(test_loader.sampler))
    test_acc = test_acc / test_num
    test_loss = test_loss / test_num

    log.write('initial_checkpoint  = %s\n' % (initial_checkpoint))
    log.write('test_acc  = %0.5f\n' % (test_acc))
    log.write('test_loss = %0.5f\n' % (test_loss[0]))
    log.write('test_num  = %d\n' % (test_num))
    log.write('\n')

    mask_average_precisions = np.array(mask_average_precisions)
    box_precisions_50 = np.array(box_precisions_50)
    log.write('-------------\n')
    log.write('mask_average_precision = %0.5f\n' % mask_average_precisions.mean())
    log.write('box_precision@0.5 = %0.5f\n' % box_precisions_50.mean())
    log.write('\n')


## evaluate post process here ####-------------------------------------
# def run_evaluate_map():
#
#     out_dir = RESULTS_DIR + '/mask-rcnn-gray-011b-drop1'
#     split   = 'valid1_ids_gray_only_43'
#
#     #------------------------------------------------------------------
#     log = Logger()
#     log.open(out_dir+'/log.evaluate.txt',mode='a')
#
#     #os.makedirs(out_dir +'/eval/'+split+'/label', exist_ok=True)
#     #os.makedirs(out_dir +'/eval/'+split+'/final', exist_ok=True)
#
#
#     image_files = glob.glob(out_dir + '/submit/npys/*.png')
#     image_files.sort()
#
#     average_precisions = []
#     for image_file in image_files:
#         #image_file = image_dir + '/0a849e0eb15faa8a6d7329c3dd66aabe9a294cccb52ed30a90c8ca99092ae732.png'
#
#         name  = image_file.split('/')[-1].replace('.png','')
#
#         image   = cv2.imread(DATA_DIR + '/image/stage1_train/' + name + '/images/' + name +'.png')
#         truth   = np.load(DATA_DIR    + '/image/stage1_train/' + name + '/multi_mask.npy').astype(np.int32)
#         predict = np.load(out_dir     + '/submit/npys/' + name + '.npy').astype(np.int32)
#         assert(predict.shape == truth.shape)
#         assert(predict.shape[:2] == image.shape[:2])
#
#
#         #image_show('image',image)
#         #image_show('mask',mask)
#         #cv2.waitKey(0)
#
#
#         #baseline labeling  -------------------------
#
#
#         # fill hole, file small, etc ...
#         # label = filter_small(label, threshold=15)
#
#
#         average_precision, precision = compute_average_precision(predict, truth)
#         average_precisions.append(average_precision)
#
#         #save and show  -------------------------
#         print(average_precision)
#
#         # overlay = (skimage.color.label2rgb(label, bg_label=0, bg_color=(0, 0, 0))*255).astype(np.uint8)
#         # cv2.imwrite(out_dir +'/eval/'+split+'/label/' + name + '.png',overlay)
#         # np.save    (out_dir +'/eval/'+split+'/label/' + name + '.npy',label)
#
#
#         # overlay1 = draw_label_contour (image, label )
#         # mask  = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
#         # final = np.hstack((image, overlay1, overlay, mask))
#         # final = final.astype(np.uint8)
#         # cv2.imwrite(out_dir +'/eval/'+split+'/final/' + name + '.png',final)
#         #
#         #
#         # image_show('image',image)
#         # image_show('mask',mask)
#         # image_show('overlay',overlay)
#         # cv2.waitKey(1)
#
#     ##----------------------------------------------
#     average_precisions = np.array(average_precisions)
#     log.write('-------------\n')
#     log.write('average_precision = %0.5f\n'%average_precisions.mean())
#     log.write('\n')
#

# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    run_evaluate()

    print('\nsucess!')
