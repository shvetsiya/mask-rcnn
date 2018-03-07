import os, sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
import pandas as pd
from train import *


## overwrite functions ###
def submit_augment(image, index):
    original_image = image.copy()
    image = resize_to_factor(image, factor=16)
    input = image.transpose([2, 0, 1])
    input = torch.from_numpy(input).float().div(255)

    return input, original_image, index


def submit_collate(batch):
    batch_size = len(batch)
    inputs = torch.stack([batch[b][0] for b in range(batch_size)], 0)
    original_images = [batch[b][1] for b in range(batch_size)]
    indices = [batch[b][2] for b in range(batch_size)]
    return [inputs, original_images, indices]


def rle_encoding(mask):
    dots = np.where(mask.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1):
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def mask_to_rles(multi_mask):
    for i in range(1, int(multi_mask.max()) + 1):
        yield rle_encoding(multi_mask == i)


def resize_multi_mask(multi_mask, desired_shape):
    result_multi_mask = np.zeros(desired_shape)
    old_w, old_h = multi_mask.shape
    new_w, new_h = desired_shape
    for x in range(new_w):
        for y in range(new_h):
            result_multi_mask[x, y] = multi_mask[round(x * old_w / new_w), round(y * old_h / new_h)]
    return result_multi_mask


def do_submit():
    out_dir = RESULTS_DIR + '/mask-rcnn-gray-011a-debug'
    initial_checkpoint = out_dir + '/checkpoint/last_model.pth'

    ## setup -----------------------------
    #os.makedirs(csv_dir, exist_ok=True)

    os.makedirs(out_dir + '/submit/overlays', exist_ok=True)
    os.makedirs(out_dir + '/submit/npys', exist_ok=True)
    #os.makedirs(out_dir +'/backup', exist_ok=True)
    #backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.submit.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir + '/log.submit.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\n')

    ## net ---------------------------------
    log.write('** net setting **\n')

    cfg = Configuration()
    net = MaskRcnnNet(cfg).cuda()
    net.load_state_dict(torch.load(initial_checkpoint))
    net.set_mode('eval')

    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
    log.write('%s\n\n' % (type(net)))

    ## dataset ---------------------------------
    log.write('** dataset setting **\n')

    test_dataset = ScienceDataset(
        #'train1_ids_gray_only1_500',
        #'valid1_ids_gray_only1_43',
        'test1_ids_65.txt',
        mode='test',
        transform=submit_augment)
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=1,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=submit_collate)
    test_num = len(test_loader.dataset)

    ## start submission here ####################################################################
    start = timer()

    predicts = []
    progress = 0
    new_test_ids = []
    rles = []
    empty_mask_ids = []
    for inputs, original_images, indices in test_loader:
        batch_size = len(indices)

        print(
            '\rpredicting: %10d/%d (%0.0f %%)  %0.2f min' %
            (progress, test_num, 100 * progress / test_num, (timer() - start) / 60),
            end='',
            flush=True)
        time.sleep(0.01)

        # forward
        inputs = Variable(inputs, volatile=True).cuda(async=True)
        net(inputs)

        ##save results ---------------------------------------
        batch_size, C, H, W = inputs.size()
        ids = test_dataset.ids

        images = inputs.data.cpu().numpy()

        masks = net.masks
        for b in range(batch_size):
            original_image = original_images[b]
            image = (images[b].transpose((1, 2, 0)) * 255).astype(np.uint8)
            image = np.clip(image.astype(np.float32) * 2.5, 0, 255)  #improve contrast

            multi_mask = resize_multi_mask(masks[b], original_image.shape[:2])

            multi_mask_overlay = multi_mask_to_overlay(
                multi_mask)  #<todo> resize to orginal image size, etc ...

            contour_overlay = cv2.resize(image.copy(),
                                         (original_image.shape[1], original_image.shape[0]))
            num_masks = int(multi_mask.max())
            for n in range(1, num_masks + 1):
                thresh = (multi_mask == n)
                contour = thresh_to_inner_contour(thresh)
                contour = contour.astype(np.float32) * 0.5
                contour_overlay = contour[:, :, np.newaxis] * np.array([0, 255, 0]) + (
                    1 - contour[:, :, np.newaxis]) * contour_overlay

            #<todo> not completed ! .....
            ##draw results ----
            # prob_overlay  = draw_multi_center(image, prob>0.4)
            #
            #
            # delta_overlay = image.copy()
            # candidates = candidates.astype(np.int32)
            # for candidate in candidates:
            #     cx,cy, minor_r,major_r, angle, score = candidate
            #     cv2.ellipse(delta_overlay, (cx,cy), (minor_r,major_r), angle, 0, 360, (255,0,255), 1)
            #
            #
            # ##draw results ---
            # nms_overlay = image.copy()
            # nms = nms.astype(np.int32)
            # for candidate in nms:
            #     cx,cy, minor_r,major_r, angle, score = candidate
            #     cv2.ellipse(nms_overlay, (cx,cy), (minor_r,major_r), angle, 0, 360, (0,255,255), 1, cv2.LINE_AA)
            #
            # original_nms_overlay = original_image.copy()
            # original_nms = original_nms.astype(np.int32)
            # for candidate in original_nms:
            #     cx,cy, minor_r,major_r, angle, score = candidate
            #     cv2.ellipse(original_nms_overlay, (cx,cy), (minor_r,major_r), angle, 0, 360, (0,255,255), 1, cv2.LINE_AA)
            #
            #
            # #--------------------------------------------
            # prob  = np.clip(prob.sum(0),0,1)
            # prob  = prob. reshape(H,W)[:, :, np.newaxis]*np.array([255,255,255])
            # all   = np.hstack((image, prob, prob_overlay, delta_overlay, nms_overlay, )).astype(np.uint8)
            #
            # image_show('all',all)

            original_size_multi_mask = resize_multi_mask(multi_mask, original_image.shape[:2])
            id = test_dataset.ids[indices[b]]
            name = id.split('/')[-1]

            if num_masks == 0:
                empty_mask_ids.append(name)
                original_size_multi_mask[0, 0] = 1

            rle = list(mask_to_rles(original_size_multi_mask))
            rles.extend(rle)
            new_test_ids.extend([name] * len(rle))

            cv2.imwrite(out_dir + '/submit/overlays/%s.multi_mask.png' % (name), multi_mask_overlay)
            cv2.imwrite(out_dir + '/submit/overlays/%s.contour.png' % (name), contour_overlay)

            # cv2.imwrite(out_dir +'/submit/overlays/%s.png'%(name),all)
            #
            # np.save(out_dir +'/submit/npys/%s.npy'%(name),multi_mask)
            # cv2.imwrite(out_dir +'/submit/npys/%s.png'%(name),label_overlay)

            # image_show('original_nms_overlay',original_nms_overlay)
            #image_show('image', image)
            #image_show('multi_mask_overlay', multi_mask_overlay)
            #image_show('contour_overlay', contour_overlay)
            #cv2.waitKey(0)

        progress += batch_size

    print('\n\nPerformed {} images'.format(progress))
    if empty_mask_ids:
        print("\nDidn't find masks for:\n{}\n".format('\n'.join(empty_mask_ids)))

    submission_df = pd.DataFrame()
    submission_df['ImageId'] = new_test_ids
    submission_df['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))

    if len(np.unique(submission_df["ImageId"])) != test_num:
        print('WARNING: Submission is not complete')
        print('Completed {} out of {}'.format(len(np.unique(submission_df["ImageId"])), test_num))
    else:
        print("Submission is complete")

    submission_df.to_csv('{}/submission_{}.cvs'.format(RESULTS_DIR, IDENTIFIER), index=False)


#--------------------------------------------------------------

# def do_submit_post_process():
#
#
#     out_dir  = RESULTS_DIR + '/unet-1cls-mask-128-00b'
#     data_dir = out_dir + '/submit'
#
#     ## start -----------------------------
#     os.makedirs(data_dir +'/predict_overlay', exist_ok=True)
#     os.makedirs(data_dir +'/final', exist_ok=True)
#
#
#
#
#     image_files = glob.glob(data_dir + '/predict_mask/*.png')
#     image_files.sort()
#
#     for image_file in image_files:
#         name = image_file.split('/')[-1].replace('.png','')
#
#         image = cv2.imread(DATA_DIR + '/stage1_test/' + name + '/images/' + name +'.png')
#         h,w,  = image.shape[:2]
#
#         mask   = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
#         thresh = (mask>128)
#
#         image_show('image',image)
#         image_show('mask',mask)
#         #cv2.waitKey(0)
#
#
#         #baseline solution -------------------------
#         predict_label = skimage.morphology.label(thresh)
#         predict_label = filter_small(predict_label, threshold=15)
#
#         np.save(data_dir +'/final/' + name + '.npy', predict_label)
#
#
#         #save and show
#         predict_overlay = (skimage.color.label2rgb(predict_label, bg_label=0, bg_color=(0, 0, 0))*255).astype(np.uint8)
#
#
#         cv2.imwrite(data_dir +'/predict_overlay/' + name + '.png',predict_overlay)
#         image_show('predict_overlay',predict_overlay)
#         cv2.waitKey(1)

# def do_merge_results():
#
#     data_dir = '/root/share/project/kaggle/science2018/results/unet-1cls-mask-128-00b/submit'
#     image_files = glob.glob(data_dir + '/predict_overlay/*.png')
#     image_files.sort()
#
#     for image_file in image_files:
#         name = image_file.split('/')[-1].replace('.png','')
#         print(name)
#
#         overlay = cv2.imread(image_file)
#
#         #megre
#         image_file1 = '/root/share/project/kaggle/science2018/results/unet-01a/submit/submission-36-fix/' + name + '.png'
#         all = cv2.imread(image_file1)
#         h,w, = all.shape[:2]
#         all[:,w//2:,:]= overlay
#         cv2.imwrite(image_file1,all)
#

# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    do_submit()
    #do_submit_post_process()
    #do_merge_results()

    print('\nsucess!')
