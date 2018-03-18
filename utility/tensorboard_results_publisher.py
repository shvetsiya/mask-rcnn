import os
from glob import glob
import numpy as np
import cv2
import matplotlib.cm
from tqdm import tqdm

from torchvision.transforms import ToTensor
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

from common import RESULTS_DIR, IDENTIFIER, DATA_DIR

OUT_DIR = RESULTS_DIR + '/mask-rcnn-50-gray500-02'


def _npy_filepath_to_image_id(filepath: str):
    return filepath.split('/')[-1].split('_')[0]


def load_nn_and_pp_masks(npys_dir, image_id):
    nn_mask_filepath = npys_dir + '/{}_nn.npy'.format(image_id)
    pp_mask_filepath = npys_dir + '/{}_pp.npy'.format(image_id)
    if len(glob(nn_mask_filepath)) == 0 or len(glob(pp_mask_filepath)) == 0:
        print('Missing files in other results:\n', nn_mask_filepath, pp_mask_filepath)
        return None, None

    nn_mask = np.load(nn_mask_filepath).astype(np.uint32)
    pp_mask = np.load(pp_mask_filepath).astype(np.uint32)

    return nn_mask, pp_mask


def apply_mask_to_image(mask, image):
    from dataset.reader import multi_mask_to_contour_overlay, multi_mask_to_color_overlay

    color_overlay = multi_mask_to_color_overlay(mask, color='brg')
    color_overlay_with_contours = multi_mask_to_contour_overlay(
        mask, color_overlay, color=[255, 255, 255])

    # image * α + mask * β + λ
    return cv2.addWeighted(color_overlay_with_contours, 0.4, image, 0.6, 0.)


def diff_masks(image, original_mask, new_mask):
    removed = (original_mask > 0) & (new_mask == 0)
    added = (original_mask == 0) & (new_mask > 0)
    common = (original_mask > 0) & (new_mask > 0)

    overlay = np.zeros(image.shape, np.uint8)
    overlay[common, 2] = 255  # blue
    overlay[added, 1] = 255  # green
    overlay[removed, 0] = 255  # red

    # image * α + mask * β + λ
    return cv2.addWeighted(overlay, 0.25, image, 0.75, 0.)


def publish_results(current_submit_npys_dir, other_submits_npys_dirs=[]):
    tb_log = SummaryWriter(OUT_DIR + '/tb_logs/submit/' + IDENTIFIER)

    current_submit_npy_files = glob(current_submit_npys_dir + '/*.npy')
    image_ids = set([_npy_filepath_to_image_id(filepath) for filepath in current_submit_npy_files])

    for image_id in tqdm(image_ids, 'Sending results to tensorboardX'):
        image_filepath = glob('{}/image/*/images/{}.png'.format(DATA_DIR, image_id))[0]
        image = cv2.imread(image_filepath, cv2.IMREAD_COLOR)

        nn_mask, pp_mask = load_nn_and_pp_masks(current_submit_npys_dir, image_id)

        results = [image, image]
        results = [apply_mask_to_image(mask, image) for mask in [nn_mask, pp_mask]]
        for other_dir in other_submits_npys_dirs:
            other_nn_mask, other_pp_mask = load_nn_and_pp_masks(other_dir, image_id)

            if other_nn_mask is None or other_pp_mask is None: continue

            results.append(diff_masks(image, original_mask=other_nn_mask, new_mask=nn_mask))
            results.append(diff_masks(image, original_mask=other_pp_mask, new_mask=pp_mask))

        stacked_results = make_grid(
            [ToTensor()(img) for img in results], normalize=True, nrow=2, pad_value=1.0)
        tb_log.add_image(image_id, stacked_results)
