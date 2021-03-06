from common import *
from utility.file import *
from utility.draw import *

from dataset.reader import *


def run_make_test_annotation():

    split = 'test1_ids_gray_only_65.txt'
    ids = read_list_from_file(DATA_DIR + '/split/' + split, comment='#')

    folder = ids[0].split('/')[0]
    os.makedirs(DATA_DIR + '/image/%s/images' % (folder), exist_ok=True)
    num_ids = len(ids)
    for i in range(num_ids):
        name = ids[i].split('/')[1]
        image_file = DATA_DIR + '/%s/%s/images/%s.png' % (folder, name, name)

        #image
        image = cv2.imread(image_file, cv2.IMREAD_COLOR)

        ## save and show -------------------------------------------
        #image_show('image', image)

        cv2.imwrite(DATA_DIR + '/image/%s/images/%s.png' % (folder, name), image)
        #cv2.waitKey(1)


def run_make_train_annotation():

    split = 'train1_ids_all_670'
    ids = read_list_from_file(DATA_DIR + '/split/' + split, comment='#')

    data_dir = DATA_DIR + '/image/stage1_train'
    os.makedirs(data_dir + '/multi_masks', exist_ok=True)
    os.makedirs(data_dir + '/overlays', exist_ok=True)
    os.makedirs(data_dir + '/images', exist_ok=True)

    num_ids = len(ids)
    for i in range(num_ids):
        id = ids[i]

        name = id.split('/')[-1]
        folder = id.split('/')[0]
        image_files = glob.glob(DATA_DIR + '/%s/%s/images/*.png' % (folder, name))
        assert (len(image_files) == 1)
        image_file = image_files[0]
        print(id)

        #image
        image = cv2.imread(image_file, cv2.IMREAD_COLOR)

        H, W, C = image.shape
        multi_mask = np.zeros((H, W), np.int32)

        mask_files = glob.glob(DATA_DIR + '/%s/%s/masks/*.png' % (folder, name))
        mask_files.sort()
        num_masks = len(mask_files)
        for i in range(num_masks):
            mask_file = mask_files[i]
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            multi_mask[np.where(mask > 128)] = i + 1

        #check
        color_overlay = multi_mask_to_color_overlay(multi_mask, color='summer')
        color1_overlay = multi_mask_to_contour_overlay(multi_mask, color_overlay, [255, 255, 255])
        contour_overlay = multi_mask_to_contour_overlay(multi_mask, image, [0, 255, 0])
        all = np.hstack((
            image,
            contour_overlay,
            color1_overlay,
        )).astype(np.uint8)

        # cv2.imwrite(data_dir +'/images/%s.png'%(name),image)

        np.save(data_dir + '/multi_masks/%s.npy' % name, multi_mask)
        cv2.imwrite(data_dir + '/multi_masks/%s.png' % name, color_overlay)
        cv2.imwrite(data_dir + '/overlays/%s.png' % name, all)
        cv2.imwrite(data_dir + '/images/%s.png' % name, image)

        # image_show('all', all)
        # cv2.waitKey(1)


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    run_make_train_annotation()
    run_make_test_annotation()

    print('sucess!')
