import os, sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
import cv2
import glob
import time
from timeit import default_timer as timer

# torch libs
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler

from common import RESULTS_DIR, IDENTIFIER, SEED, PROJECT_PATH, ALL_TEST_IMAGE_ID

from utility.file import Logger
from net.resnet50_mask_rcnn.configuration import Configuration
from net.resnet50_mask_rcnn.model import MaskRcnnNet
from net.metric import run_length_encode
from dataset.reader import ScienceDataset, multi_mask_to_contour_overlay, \
        multi_mask_to_color_overlay
from dataset.transform import pad_to_factor


def _revert(net, images):
    """Reverts test-time-augmentation (e.g., unpad, scale back to input image size, etc).
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
    for b in range(batch_size):
        image = images[b]
        height, width = image.shape[:2]

        index = (net.detections[:, 0] == b).nonzero().view(-1)
        net.detections = torch_clip_proposals(net.detections, index, width, height)

        net.masks[b] = net.masks[b][:height, :width]


def _submit_augment(image, index):
    pad_image = pad_to_factor(image, factor=16)
    input = torch.from_numpy(pad_image.transpose((2, 0, 1))).float().div(255)
    return input, image, index


def _submit_collate(batch):
    batch_size = len(batch)
    inputs = torch.stack([batch[b][0] for b in range(batch_size)], 0)
    images = [batch[b][1] for b in range(batch_size)]
    indices = [batch[b][2] for b in range(batch_size)]

    return [inputs, images, indices]


def run_submit():
    OUT_DIR = RESULTS_DIR + '/mask-rcnn-50-gray500-02'
    initial_checkpoint = RESULTS_DIR + '/mask-rcnn-50-gray500-02/checkpoint/best_model.pth'

    os.makedirs(OUT_DIR + '/submit/overlays', exist_ok=True)
    os.makedirs(OUT_DIR + '/submit/npys', exist_ok=True)
    os.makedirs(OUT_DIR + '/checkpoint', exist_ok=True)

    log = Logger()
    log.open(OUT_DIR + '/log.evaluate.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % OUT_DIR)
    log.write('\n')

    cfg = Configuration()
    net = MaskRcnnNet(cfg).cuda()

    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(
            torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    log.write('%s\n\n' % (type(net)))
    log.write('\n')

    log.write('** dataset setting **\n')

    test_dataset = ScienceDataset(
        #'train1_ids_gray_only1_500',
        #'valid1_ids_gray_only1_43',
        #'debug1_ids_gray_only_10',
        'test1_ids_gray_only_65.txt',
        mode='test',
        transform=_submit_augment)
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=1,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=_submit_collate)

    log.write('\ttest_dataset.split = %s\n' % (test_dataset.split))
    log.write('\tlen(test_dataset)  = %d\n' % (len(test_dataset)))
    log.write('\n')

    ## start evaluation here! ##############################################
    log.write('** start evaluation here! **\n')
    start = timer()

    test_num = len(test_loader.dataset)
    for i, (inputs, images, indices) in enumerate(test_loader, 0):
        print(
            '\rpredicting: %10d/%d (%0.0f %%)  %0.2f min' %
            (i, test_num - 1, 100 * i / (test_num - 1), (timer() - start) / 60),
            end='',
            flush=True)
        time.sleep(0.01)

        net.set_mode('test')
        with torch.no_grad():
            inputs = Variable(inputs).cuda()
            net(inputs)
            # Resize results to original images shapes.
            _revert(net, images)

        batch_size = inputs.size()[0]
        # NOTE: Current version support batch_size==1 for variable size input. To use
        # batch_size > 1, need to fix code for net.windows, etc.
        assert (batch_size == 1)

        inputs = inputs.data.cpu().numpy()

        window = net.rpn_window
        rpn_logits_flat = net.rpn_logits_flat.data.cpu().numpy()
        rpn_deltas_flat = net.rpn_deltas_flat.data.cpu().numpy()
        detections = net.detections
        masks = net.masks

        for b in range(batch_size):
            image = images[b]
            mask = masks[b]

            contour_overlay = multi_mask_to_contour_overlay(mask, image, color=[0, 255, 0])
            color_overlay = multi_mask_to_color_overlay(mask, color='summer')
            color1_overlay = multi_mask_to_contour_overlay(
                mask, color_overlay, color=[255, 255, 255])

            all = np.hstack((image, contour_overlay, color1_overlay))

            # --------------------------------------------
            id = test_dataset.ids[indices[b]]
            name = id.split('/')[-1]

            np.save(OUT_DIR + '/submit/npys/%s.npy' % (name), mask)
            cv2.imwrite(OUT_DIR + '/submit/overlays/%s.png' % (name), all)

            os.makedirs(OUT_DIR + '/submit/psds/%s' % name, exist_ok=True)
            cv2.imwrite(OUT_DIR + '/submit/psds/%s/%s.png' % (name, name), image)
            cv2.imwrite(OUT_DIR + '/submit/psds/%s/%s.mask.png' % (name, name), color_overlay)
            cv2.imwrite(OUT_DIR + '/submit/psds/%s/%s.contour.png' % (name, name), contour_overlay)

    assert (test_num == len(test_loader.sampler))

    log.write('initial_checkpoint  = %s\n' % (initial_checkpoint))
    log.write('test_num  = %d\n' % (test_num))
    log.write('\n')


def filter_small(multi_mask, threshold):
    num_masks = int(multi_mask.max())

    j = 0
    for i in range(num_masks):
        thresh = (multi_mask == (i + 1))

        area = thresh.sum()
        if area < threshold:
            multi_mask[thresh] = 0
        else:
            multi_mask[thresh] = (j + 1)
            j = j + 1

    return multi_mask


def shrink_by_one(multi_mask):

    multi_mask1 = np.zeros(multi_mask.shape, np.int32)

    num = int(multi_mask.max())
    for m in range(num):
        mask = (multi_mask == m + 1)
        contour = mask_to_inner_contour(mask)
        thresh = thresh & (~contour)
        multi_mask1[thresh] = m + 1

    return multi_mask1


def run_npy_to_sumbit_csv():

    image_dir = '../image/stage1_test/images'
    submit_dir = '../results/mask-rcnn-50-gray500-02/submit'

    npy_dir = submit_dir + '/npys'
    csv_file = submit_dir + '/submission-gray53-only.csv'

    ## start -----------------------------
    all_num = 0
    cvs_ImageId = []
    cvs_EncodedPixels = []

    npy_files = glob.glob(npy_dir + '/*.npy')
    for npy_file in npy_files:
        name = npy_file.split('/')[-1].replace('.npy', '')

        multi_mask = np.load(npy_file)

        #<todo> ---------------------------------
        #post process here
        multi_mask = filter_small(multi_mask, 8)
        #<todo> ---------------------------------

        num = int(multi_mask.max())
        for m in range(num):
            rle = run_length_encode(multi_mask == m + 1)
            cvs_ImageId.append(name)
            cvs_EncodedPixels.append(rle)
        all_num += num

        #<debug> ------------------------------------
        print(all_num, num)  ##GT is 4152?
        image_file = image_dir + '/%s.png' % name
        image = cv2.imread(image_file)
        color_overlay = multi_mask_to_color_overlay(multi_mask)
        color1_overlay = multi_mask_to_contour_overlay(multi_mask, color_overlay)
        contour_overlay = multi_mask_to_contour_overlay(multi_mask, image, [0, 255, 0])
        #all = np.hstack((image, contour_overlay, color1_overlay)).astype(np.uint8)
        #image_show('all',all)
        #cv2.waitKey(1)

    #exit(0)
    # submission csv  ----------------------------

    # kaggle submission requires all test image to be listed!
    for t in ALL_TEST_IMAGE_ID:
        cvs_ImageId.append(t)
        cvs_EncodedPixels.append('')  #null

    df = pd.DataFrame({'ImageId': cvs_ImageId, 'EncodedPixels': cvs_EncodedPixels})
    df.to_csv(csv_file, index=False, columns=['ImageId', 'EncodedPixels'])


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    run_submit()
    run_npy_to_sumbit_csv()

    print('\nsucess!')
