import os, sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
import cv2

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
import matplotlib.pyplot as plt

from utility.file import Logger
from net.resnet50_mask_rcnn.configuration import Configuration
from net.resnet50_mask_rcnn.draw import draw_multi_proposal_metric, draw_mask_metric, image_show
from net.resnet50_mask_rcnn.model import MaskRcnnNet
from net.metric import compute_average_precision_for_mask, compute_precision_for_box, HIT
from dataset.reader import ScienceDataset, multi_mask_to_annotation, instance_to_multi_mask, \
        multi_mask_to_contour_overlay, multi_mask_to_color_overlay
from dataset.transform import pad_to_factor


class Evaluator(object):

    def __init__(self):
        self.OUT_DIR = RESULTS_DIR + '/mask-rcnn-50-gray500-02'
        self.OVERLAYS_DIR = self.OUT_DIR + '/evaluate/overlays'
        self.STATS_DIR = self.OUT_DIR + '/evaluate/stats'
        self.logger = Logger()

        ## setup  ---------------------------
        os.makedirs(self.OVERLAYS_DIR, exist_ok=True)
        os.makedirs(self.STATS_DIR, exist_ok=True)
        os.makedirs(self.OUT_DIR + '/evaluate/npys', exist_ok=True)
        os.makedirs(self.OUT_DIR + '/checkpoint', exist_ok=True)
        os.makedirs(self.OUT_DIR + '/backup', exist_ok=True)

        logger = self.logger
        logger.open(self.OUT_DIR + '/log.evaluate.txt', mode='a')
        logger.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
        logger.write('** some experiment setting **\n')
        logger.write('\tSEED         = %u\n' % SEED)
        logger.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
        logger.write('\tOUT_DIR      = %s\n' % self.OUT_DIR)
        logger.write('\n')

        ## dataset ----------------------------------------
        logger.write('** dataset setting **\n')

        self.test_dataset = ScienceDataset(
            'train1_ids_gray2_500',
            # 'valid1_ids_gray2_43',
            mode='train',
            #'debug1_ids_gray2_10', mode='train',
            transform=self._eval_augment)
        self.test_loader = DataLoader(
            self.test_dataset,
            sampler=SequentialSampler(self.test_dataset),
            batch_size=1,
            drop_last=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=self._eval_collate)

        logger.write('\ttest_dataset.split = %s\n' % (self.test_dataset.split))
        logger.write('\tlen(self.test_dataset)  = %d\n' % (len(self.test_dataset)))
        logger.write('\n')

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

    def _save_prediction_png(self, name: str, mask, detection, truth_box, truth_label,
                             truth_instance, image):
        cfg = self.cfg

        contour_overlay = multi_mask_to_contour_overlay(mask, image=image, color=[0, 255, 0])
        color_overlay = multi_mask_to_color_overlay(mask, color='summer')
        color_overlay_with_contour = multi_mask_to_contour_overlay(
            mask, image=color_overlay, color=[255, 255, 255])

        all1 = np.hstack((image, contour_overlay, color_overlay_with_contour))
        all6 = draw_multi_proposal_metric(cfg, image, detection, truth_box, truth_label,
                                          [0, 255, 255], [255, 0, 255], [255, 255, 0])
        all7 = draw_mask_metric(cfg, image, mask, truth_box, truth_label, truth_instance)

        cv2.imwrite('{}/{}_all1.png'.format(self.OVERLAYS_DIR, name), all1)
        cv2.imwrite('{}/{}_all6.png'.format(self.OVERLAYS_DIR, name), all6)
        cv2.imwrite('{}/{}_all7.png'.format(self.OVERLAYS_DIR, name), all7)

    def _append_hit_and_miss_stats(self, name: str, truth_boxes, truth_box_results, thresholds,
                                   bb_results: pd.DataFrame):
        """Checks which ground thruth boxes are hit and which are missed for each threshold level.
        Populates results dataframe.

        Args:
            truth_boxes: an array of ground truth boxes.
            thruth_box_results: an list of results for each threshold. Each result is an array of
                truth_boxes length. Each element of the array is whether HIT or not.
            thresholds: threshold levels for IoU.
            bb_results: bounding boxes results.
        """
        for threshold_index, threshold in enumerate(thresholds):
            for box, result in zip(truth_boxes, truth_box_results[threshold_index]):
                x0, y0, x1, y1 = box.astype(np.int32)
                w, h = (x1 - x0, y1 - y0)
                bb_results.loc[bb_results.shape[0]] = {
                    'id': name,
                    'w': w,
                    'h': h,
                    'threshold': threshold,
                    'is_hit': result == HIT
                }

    def _append_results_stats(self, name: str, box_precision, thresholds, mask_average_precision,
                              overall_results):
        """Appends overall precision results to the results dataframe.
        """
        results_row = {'id': name, 'mask_average_precision': mask_average_precision}
        for threshold_index, threshold in enumerate(thresholds):
            results_row['box_precision_{}'.format(int(
                threshold * 100))] = box_precision[threshold_index]
        overall_results.loc[overall_results.shape[0]] = results_row

    def run_evaluate(self, model_checkpoint):
        self.cfg = Configuration()

        logger = self.logger
        thresholds = [0.5, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

        # TODO(alexander): Populate this.
        overall_results_columns = ['id', 'mask_average_precision']
        for threshold in thresholds:
            overall_results_columns.append('box_precision_{}'.format(int(threshold * 100)))
        overall_results = pd.DataFrame(columns=overall_results_columns)

        bb_results = pd.DataFrame(columns=['id', 'w', 'h', 'threshold', 'is_hit'])

        logger.write('\tmodel checkpoint = %s\n' % model_checkpoint)
        net = MaskRcnnNet(self.cfg).cuda()
        net.load_state_dict(torch.load(model_checkpoint, map_location=lambda storage, loc: storage))
        net.set_mode('test')
        logger.write('\tmodel type: %s\n\n' % (type(net)))

        logger.write('** starting evaluation! **\n\n')
        logger.write('index | id | mask_average_precision (box_precision)\n')
        mask_average_precisions = []
        box_precisions_50 = []

        hit_box_sizes = dict()
        missed_box_sizes = dict()
        for threshold in thresholds:
            hit_box_sizes[threshold] = []
            missed_box_sizes[threshold] = []

        test_num = 0
        for inputs, truth_boxes, truth_labels, truth_instances, metas, images, indices in self.test_loader:
            if all((truth_label > 0).sum() == 0 for truth_label in truth_labels):
                continue

            with torch.no_grad():
                inputs = Variable(inputs).cuda()
                net(inputs, truth_boxes, truth_labels, truth_instances)

            # Resize results to original images shapes.
            self._revert(net, images)

            batch_size = inputs.size()[0]
            # NOTE: Current version support batch_size==1 for variable size input. To use
            # batch_size > 1, need to fix code for net.windows, etc.
            assert (batch_size == 1)

            inputs = inputs.data.cpu().numpy()
            masks = net.masks
            detections = net.detections.cpu().numpy()

            for index_in_batch in range(batch_size):
                test_num += 1

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
                    mask, truth_mask, t_range=thresholds)

                box_precision, box_recall, box_result, truth_box_result = \
                    compute_precision_for_box(box, truth_box, truth_label, thresholds)

                mask_average_precisions.append(mask_average_precision)
                box_precisions_50.append(box_precision[0])

                id = self.test_dataset.ids[indices[index_in_batch]]
                name = id.split('/')[-1]

                self._append_hit_and_miss_stats(name, truth_box, truth_box_result, thresholds,
                                                bb_results)
                self._append_results_stats(name, box_precision, thresholds, mask_average_precision,
                                           overall_results)
                self._save_prediction_png(
                    name,
                    mask=mask,
                    detection=detection,
                    truth_box=truth_box,
                    truth_label=truth_label,
                    truth_instance=truth_instance,
                    image=image)

                print('%d\t%s\t%0.5f  (%0.5f)' % (test_num, name, mask_average_precision,
                                                  box_precision[0]))

        for threshold in thresholds:
            hit_results = bb_results[(bb_results.is_hit == True) &
                                     (bb_results.threshold == threshold)]
            miss_results = bb_results[(bb_results.is_hit == False) &
                                      (bb_results.threshold == threshold)]
            plt.plot(hit_results.w, hit_results.h, 'bo', miss_results.w, miss_results.h, 'ro')
            plt.savefig('{}/hits_and_misses_{}.png'.format(self.STATS_DIR, int(threshold * 100)))

        bb_results.to_csv(self.STATS_DIR + '/bb_results.csv')
        overall_results.to_csv(self.STATS_DIR + '/overall_results.csv')

        mask_average_precisions = np.array(mask_average_precisions)
        box_precisions_50 = np.array(box_precisions_50)
        logger.write('-------------\n')
        logger.write('mask_average_precision = %0.5f\n' % mask_average_precisions.mean())
        logger.write('box_precision@0.5 = %0.5f\n' % box_precisions_50.mean())
        logger.write('\n')


if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    model_checkpoint = RESULTS_DIR + '/mask-rcnn-50-gray500-02/checkpoint/00008500_model.pth'
    evaluator = Evaluator()
    evaluator.run_evaluate(model_checkpoint)
