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


class Evaluator(object):

    def _revert(self, net: MaskRcnnNet, images: list):
        """Adjusts the net results to original images sizes.

        returns:
            nothing
        """

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

        batch_size = len(images)
        for index_in_batch in range(batch_size):
            image = images[index_in_batch]
            height, width = image.shape[:2]

            index = (net.detections[:, 0] == index_in_batch).nonzero().view(-1)
            net.detections = torch_clip_proposals(net.detections, index, width, height)
            net.masks[index_in_batch] = net.masks[index_in_batch][:height, :width]

    def _eval_augment(self, image, multi_mask, meta, index):
        pad_image = pad_to_factor(image, factor=16)
        input = torch.from_numpy(pad_image.transpose((2, 0, 1))).float().div(255)
        box, label, instance = multi_mask_to_annotation(multi_mask)

        return input, box, label, instance, meta, image, index

    def _eval_collate(self, batch):
        batch_size = len(batch)
        #for index_in_batch in range(batch_size): print (batch[index_in_batch][0].size())
        inputs = torch.stack([batch[index_in_batch][0] for index_in_batch in range(batch_size)], 0)
        boxes = [batch[index_in_batch][1] for index_in_batch in range(batch_size)]
        labels = [batch[index_in_batch][2] for index_in_batch in range(batch_size)]
        instances = [batch[index_in_batch][3] for index_in_batch in range(batch_size)]
        metas = [batch[index_in_batch][4] for index_in_batch in range(batch_size)]
        images = [batch[index_in_batch][5] for index_in_batch in range(batch_size)]
        indices = [batch[index_in_batch][6] for index_in_batch in range(batch_size)]

        return [inputs, boxes, labels, instances, metas, images, indices]

    def run_evaluate(self):
        out_dir = RESULTS_DIR + '/mask-rcnn-50-gray500-02'
        initial_checkpoint = \
            RESULTS_DIR + '/mask-rcnn-50-gray500-02/checkpoint/00008500_model.pth'

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

        cfg = Configuration()
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
            transform=self._eval_augment)
        test_loader = DataLoader(
            test_dataset,
            sampler=SequentialSampler(test_dataset),
            batch_size=1,
            drop_last=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=self._eval_collate)

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

            # Resize results to original images shapes.
            self._revert(net, images)

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

            for index_in_batch in range(batch_size):
                #image0 = (inputs[index_in_batch].transpose((1,2,0))*255).astype(np.uint8)
                image = images[index_in_batch]
                height, width = image.shape[:2]
                mask = masks[index_in_batch]

                index = np.where(detections[:, 0] == index_in_batch)[0]
                detection = detections[index]
                box = detection[:, 1:5]

                truth_mask = instance_to_multi_mask(truth_instances[index_in_batch])
                truth_box = truth_boxes[index_in_batch]
                truth_label = truth_labels[index_in_batch]
                truth_instance = truth_instances[index_in_batch]

                mask_average_precision, mask_precision = compute_average_precision_for_mask(
                    mask, truth_mask, t_range=np.arange(0.5, 1.0, 0.05))

                box_precision, box_recall, box_result, truth_box_result = \
                    compute_precision_for_box(box, truth_box, truth_label, threshold=[0.5])
                box_precision = box_precision[0]

                mask_average_precisions.append(mask_average_precision)
                box_precisions_50.append(box_precision)

                # --------------------------------------------
                id = test_dataset.ids[indices[index_in_batch]]
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


if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
    evaluator = Evaluator()
    evaluator.run_evaluate()
    print('\nsucess!')
