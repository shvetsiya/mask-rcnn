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
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid

from tensorboardX import SummaryWriter

from common import RESULTS_DIR, IDENTIFIER, SEED, PROJECT_PATH, ALL_TEST_IMAGE_ID, DATA_DIR

from utility.file import Logger
from net.resnet50_mask_rcnn.configuration import Configuration
from net.resnet50_mask_rcnn.model import MaskRcnnNet
from net.metric import run_length_encode
from dataset.reader import ScienceDataset, multi_mask_to_contour_overlay, \
        multi_mask_to_color_overlay
from dataset.transform import pad_to_factor
from postprocessing.utils import post_process
from utility.tensorboard_results_publisher import publish_results

OUT_DIR = RESULTS_DIR + '/mask-rcnn-50-gray500-02'


def _revert(results, images):
    """Reverts test-time-augmentation (e.g., unpad, scale back to input image size, etc).
    """

    assert len(results) == len(images), 'Results and images should be the same length'

    batch_size = len(images)
    for index_in_batch in range(batch_size):
        result = results[index_in_batch]
        image = images[index_in_batch]
        height, width = image.shape[:2]

        result.multi_mask = result.multi_mask[:height, :width]
        for bounding_box in result.bounding_boxes:
            x0, y0, x1, y1 = bounding_box.coordinates
            x0, x1 = min((x0, x1), (height, height))
            y0, y1 = min((y0, y1), (width, width))
            bounding_box.coordinates = (x0, y0, x1, y1)


def _submit_augment(image, index):
    pad_image = pad_to_factor(image, factor=16)
    input = torch.from_numpy(pad_image.transpose((2, 0, 1))).float().div(255)
    return input, image, index


def _submit_collate(batch):
    batch_size = len(batch)
    inputs = torch.stack([batch[index_in_batch][0] for index_in_batch in range(batch_size)], 0)
    images = [batch[index_in_batch][1] for index_in_batch in range(batch_size)]
    indices = [batch[index_in_batch][2] for index_in_batch in range(batch_size)]

    return [inputs, images, indices]


def run_multi_masks_prediction():
    initial_checkpoint = RESULTS_DIR + '/mask-rcnn-50-gray500-02/checkpoint/best_model.pth'

    os.makedirs(OUT_DIR + '/submit/overlays', exist_ok=True)
    os.makedirs(OUT_DIR + '/submit/npys', exist_ok=True)

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

    for inputs, images, indices in tqdm(test_loader, 'Mask-RCNN predictions'):
        batch_size = inputs.size()[0]
        # NOTE: Current version support batch_size==1 for variable size input. To use
        # batch_size > 1, need to fix code for net.windows, etc.
        assert (batch_size == 1)

        with torch.no_grad():
            inputs = Variable(inputs).cuda()
            net(inputs)

        # Resize results to original images shapes.
        results = net.results
        _revert(results, images)

        for index_in_batch in range(batch_size):
            image = images[index_in_batch]
            index = indices[index_in_batch]
            mask = results[index_in_batch].multi_mask

            image_id = test_dataset.ids[index]

            save_prediction_info(image_id, image, mask)


def save_prediction_info(image_id: str, image: np.array, mask: np.array):
    contour_overlay = multi_mask_to_contour_overlay(mask, image, color=[0, 255, 0])
    color_overlay = multi_mask_to_color_overlay(mask, color='brg')
    color_overlay_with_contours = multi_mask_to_contour_overlay(
        mask, color_overlay, color=[255, 255, 255])

    stacked_results = np.hstack((image, contour_overlay, color_overlay_with_contours))

    name = image_id.split('/')[-1]

    np.save(OUT_DIR + '/submit/npys/%s_nn.npy' % (name), mask)
    cv2.imwrite(OUT_DIR + '/submit/overlays/%s.png' % (name), stacked_results)

    os.makedirs(OUT_DIR + '/submit/psds/%s' % name, exist_ok=True)
    cv2.imwrite(OUT_DIR + '/submit/psds/%s/%s.png' % (name, name), image)
    cv2.imwrite(OUT_DIR + '/submit/psds/%s/%s.mask.png' % (name, name), color_overlay)
    cv2.imwrite(OUT_DIR + '/submit/psds/%s/%s.contour.png' % (name, name), contour_overlay)


def shrink_by_one(multi_mask):
    multi_mask1 = np.zeros(multi_mask.shape, np.int32)

    num = int(multi_mask.max())
    for m in range(num):
        mask = (multi_mask == m + 1)
        contour = mask_to_inner_contour(mask)
        thresh = thresh & (~contour)
        multi_mask1[thresh] = m + 1

    return multi_mask1


def run_post_processing():
    image_dir = '../image/stage1_test/images'
    submit_dir = '../results/mask-rcnn-50-gray500-02/submit'

    npy_dir = submit_dir + '/npys'
    csv_file = submit_dir + '/submission-gray53-only.csv'

    image_ids = []
    encoded_pixels = []

    npy_files = glob.glob(npy_dir + '/*_nn.npy')
    for npy_file in tqdm(npy_files, 'Postprocessing'):
        name = npy_file.split('/')[-1].replace('_nn.npy', '')

        nn_multi_mask = np.load(npy_file).astype(np.uint32)
        pp_multi_mask = post_process(nn_multi_mask)

        np.save(OUT_DIR + '/submit/npys/%s_pp.npy' % (name), pp_multi_mask)

        for color in range(1, pp_multi_mask.max() + 1):
            rle = run_length_encode(pp_multi_mask == color)
            image_ids.append(name)
            encoded_pixels.append(rle)

    # NOTE: Kaggle submission requires all test image to be listed.
    for t in ALL_TEST_IMAGE_ID:
        image_ids.append(t)
        encoded_pixels.append('')

    df = pd.DataFrame({'ImageId': image_ids, 'EncodedPixels': encoded_pixels})
    df.to_csv(csv_file, index=False, columns=['ImageId', 'EncodedPixels'])


if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    best_submit_npys_dir = '../results/mask-rcnn-50-gray500-02/submit_432/npys'
    submit_dir = '../results/mask-rcnn-50-gray500-02/submit'

    # run_multi_masks_prediction()
    run_post_processing()
    publish_results(submit_dir + '/npys', [best_submit_npys_dir])

    print('Sucess!')
