import os, sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
import cv2
import glob
import time
from tqdm import tqdm
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

OUT_DIR = RESULTS_DIR + '/mask-rcnn-50-gray500-02'


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

    test_dataset = ScienceDataset('test1_ids_gray_only_53', mode='test', transform=_submit_augment)
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

    log.write('** start evaluation here! **\n')

    net.set_mode('test')

    for inputs, images, indices in tqdm(test_loader):
        batch_size = inputs.size()[0]
        # NOTE: Current version support batch_size==1 for variable size input. To use
        # batch_size > 1, need to fix code for net.windows, etc.
        assert (batch_size == 1)

        with torch.no_grad():
            inputs = Variable(inputs).cuda()
            net(inputs)
            # Resize results to original images shapes.
            _revert(net, images)

        detections = net.detections
        masks = net.masks

        for b in range(batch_size):
            image = images[b]
            mask = masks[b].astype(np.uint8)
            index = indices[b]

            image_id = test_dataset.ids[index]
            box_indices = np.where(detections[:, 0] == b)[0]
            boxes = detections[box_indices, 1:5]

            save_prediction_info(image_id, image, mask)


def save_prediction_info(image_id: str, image: np.array, mask: np.array):
    contour_overlay = multi_mask_to_contour_overlay(mask, image, color=[0, 255, 0])
    color_overlay = multi_mask_to_color_overlay(mask, color='summer')
    color1_overlay = multi_mask_to_contour_overlay(mask, color_overlay, color=[255, 255, 255])

    all = np.hstack((image, contour_overlay, color1_overlay))

    name = image_id.split('/')[-1]

    np.save(OUT_DIR + '/submit/npys/%s.npy' % (name), mask)
    cv2.imwrite(OUT_DIR + '/submit/overlays/%s.png' % (name), all)

    os.makedirs(OUT_DIR + '/submit/psds/%s' % name, exist_ok=True)
    cv2.imwrite(OUT_DIR + '/submit/psds/%s/%s.png' % (name, name), image)
    cv2.imwrite(OUT_DIR + '/submit/psds/%s/%s.mask.png' % (name, name), color_overlay)
    cv2.imwrite(OUT_DIR + '/submit/psds/%s/%s.contour.png' % (name, name), contour_overlay)


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

    image_ids = []
    encoded_pixels = []

    npy_files = glob.glob(npy_dir + '/*.npy')
    for npy_file in npy_files:
        name = npy_file.split('/')[-1].replace('.npy', '')

        multi_mask = np.load(npy_file)

        #<todo> ---------------------------------
        #post process here
        multi_mask = filter_small(multi_mask, 8)
        #<todo> ---------------------------------

        for color in range(1, multi_mask.max() + 1):
            rle = run_length_encode(multi_mask == color)
            image_ids.append(name)
            encoded_pixels.append(rle)

    # NOTE: Kaggle submission requires all test image to be listed.
    for t in ALL_TEST_IMAGE_ID:
        image_ids.append(t)
        encoded_pixels.append('')  #null

    df = pd.DataFrame({'ImageId': image_ids, 'EncodedPixels': encoded_pixels})
    df.to_csv(csv_file, index=False, columns=['ImageId', 'EncodedPixels'])


if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    run_submit()
    run_npy_to_sumbit_csv()

    print('Sucess!')
