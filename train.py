import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #'3,2,1,0'

from common import *
from utility.file import *

from dataset.reader import *

from net.rate import *
# from net.loss import *
# from net.process import *
# from net.metric import *

WIDTH, HEIGHT = 128, 128

# -------------------------------------------------------------------------------------
from model.mask_rcnn_resnet50_fpn_net import *


def train_augment(image, multi_mask, index):

    image, multi_mask = random_crop_transform2(image, multi_mask, WIDTH, HEIGHT)

    #image = random_brightness_shift_transform(image, limit=[0.2,0.5], u=0.5)
    #image = random_mask_noise_transform(image, limit=[0, 0.1], u=0.5)
    #image = random_contrast_transform(image, limit=[0.5,1.5], u=0.5)
    #---------------------------------------
    box, label, instance = multi_mask_to_annotation(multi_mask)

    input = image.transpose((2, 0, 1))
    input = torch.from_numpy(input).float().div(255)

    return input, box, label, instance, index


def valid_augment(image, multi_mask, index):

    image, multi_mask = fix_crop_transform2(image, multi_mask, -1, -1, WIDTH, HEIGHT)

    #---------------------------------------
    box, label, instance = multi_mask_to_annotation(multi_mask)

    input = image.transpose([2, 0, 1])
    input = torch.from_numpy(input).float().div(255)

    return input, box, label, instance, index


def train_collate(batch):

    batch_size = len(batch)
    #for b in range(batch_size): print (batch[b][0].size())
    inputs = torch.stack([batch[b][0] for b in range(batch_size)], 0)
    boxes = [batch[b][1] for b in range(batch_size)]
    labels = [batch[b][2] for b in range(batch_size)]
    instances = [batch[b][3] for b in range(batch_size)]
    indices = [batch[b][4] for b in range(batch_size)]

    return [inputs, boxes, labels, instances, indices]


### debug and draw #########################################################


def debug_and_draw(net, inputs, truth_boxes, truth_labels, truth_instances, mode='test'):

    mode0 = net.mode
    net.set_mode(mode)

    if mode in ['test']:
        inputs = Variable(inputs.data, volatile=True).cuda()
        net(inputs)

    elif mode in ['train', 'valid']:
        net(inputs, truth_boxes, truth_labels, truth_instances)
    #loss = net.loss( inputs, truth_boxes, truth_labels, truth_instances )

    #
    #net( tensors, labels,  gt_boxes  )
    # =====================================================
    batch_size, C, H, W = inputs.size()
    images = inputs.data.cpu().numpy()
    rpn_probs_flat = net.rpn_probs_flat.data.cpu().numpy()
    rpn_deltas_flat = net.rpn_deltas_flat.data.cpu().numpy()
    windows = net.rpn_windows

    proposals = net.rpn_proposals.data.cpu().numpy()
    detections = net.detections
    masks = net.masks

    #print('train',batch_size)
    for b in range(batch_size):
        image = (images[b].transpose((1, 2, 0)) * 255).astype(np.uint8)
        image = np.clip(image.astype(np.float32) * 2.5, 0, 255)  #improve contrast
        contour_overlay = image.copy()

        label = truth_labels[b]
        prob = rpn_probs_flat[b]
        delta = rpn_deltas_flat[b]

        image_rpn_proposal_before_nms = draw_rpn_proposal_before_nms(image, prob, delta, windows,
                                                                     0.95)

        proposal = proposals[np.where(proposals[:, 0] == b)]
        image1 = image.copy()  #draw_label_as_gt_boxes(image, label)
        image_rpn_proposal_after_nms = draw_rpn_proposal_after_nms(image1, proposal, top=100000)

        detection = detections[b]
        image_rcnn_detection_nms = draw_rcnn_detection_nms(image, detection, threshold=0.5)
        #print(len(detection))

        multi_mask = masks[b]
        multi_mask_overlay = multi_mask_to_overlay(multi_mask)

        num_masks = int(multi_mask.max())
        for n in range(num_masks):
            thresh = multi_mask == n
            contour = thresh_to_inner_contour(thresh)
            contour = contour.astype(np.float32) * 0.5
            contour_overlay = contour[:, :, np.newaxis] * np.array(
                (0, 255, 0)) + (1 - contour[:, :, np.newaxis]) * contour_overlay

        #image_show('mask',mask/mask.max()*255,2)

        all = np.hstack((image, image_rpn_proposal_before_nms, image_rpn_proposal_after_nms,
                         image_rcnn_detection_nms, multi_mask_overlay, contour_overlay))
        #cv2.imwrite(out_dir +'/train/%05d.png'%indices[b],all)
        #image_show('all', all, 2)
        #cv2.waitKey(0)

    net.set_mode(mode0)


### training ##############################################################
def evaluate(net, test_loader):

    test_num = 0
    test_loss = np.zeros(6, np.float32)
    test_acc = 0
    for i, (inputs, boxes, labels, instances, indices) in enumerate(test_loader, 0):
        inputs = Variable(inputs, volatile=True).cuda()

        net(inputs, boxes, labels, instances)
        loss = net.loss(inputs, boxes, labels, instances)

        # acc    = dice_loss(masks, labels) #todo

        batch_size = len(indices)
        test_acc += 0  #batch_size*acc[0][0]
        test_loss += batch_size * np.array((
            loss.cpu().data.numpy()[0],
            net.rpn_cls_loss.cpu().data.numpy()[0],
            net.rpn_reg_loss.cpu().data.numpy()[0],
            net.rcnn_cls_loss.cpu().data.numpy()[0],
            net.rcnn_reg_loss.cpu().data.numpy()[0],
            net.mask_cls_loss.cpu().data.numpy()[0],
        ))
        test_num += batch_size

    assert (test_num == len(test_loader.sampler))
    test_acc = test_acc / test_num
    test_loss = test_loss / test_num
    return test_loss, test_acc


#--------------------------------------------------------------
def run_train():

    out_dir = RESULTS_DIR + '/mask-rcnn-gray-011a-debug'
    initial_checkpoint = RESULTS_DIR + '/mask-rcnn-gray-011a-debug/checkpoint/00012600_model.pth'
    #

    pretrain_file = None  #imagenet pretrain
    ## setup  -----------------
    os.makedirs(out_dir + '/checkpoint', exist_ok=True)
    os.makedirs(out_dir + '/train', exist_ok=True)
    #os.makedirs(out_dir +'/backup', exist_ok=True)
    #backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.train.%s.zip'%IDENTIFIER)

    log = Logger()
    log.open(out_dir + '/log.train.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')

    ## net ----------------------
    log.write('** net setting **\n')
    cfg = Configuration()
    net = MaskRcnnNet(cfg).cuda()

    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(
            torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    elif pretrain_file is not None:
        log.write('\tpretrained_file = %s\n' % pretrain_file)
        #load_pretrain_file(net, pretrain_file)

    log.write('%s\n\n' % (type(net)))
    log.write('\n')

    ## optimiser ----------------------------------
    iter_accum = 1
    batch_size = 4  ##NUM_CUDA_DEVICES*512 #256//iter_accum #512 #2*288//iter_accum

    num_iters = 1000 * 1000
    iter_smooth = 20
    iter_log = 50
    iter_valid = 100
    iter_save = [0, num_iters - 1] + list(range(0, num_iters, 100))  #1*1000

    LR = None  #LR = StepLR([ (0, 0.01),  (200, 0.001),  (300, -1)])
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=0.01 / iter_accum,
        momentum=0.9,
        weight_decay=0.0001)

    start_iter = 0
    start_epoch = 0.
    if initial_checkpoint is not None:
        #checkpoint = torch.load(initial_checkpoint.replace('_model.pth', '_optimizer.pth'))
        #checkpoint = torch.load(initial_checkpoint)
        start_iter = 0  #checkpoint['iter']
        start_epoch = 0  #checkpoint['epoch']
        #optimizer.load_state_dict(checkpoint['optimizer'])

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    train_dataset = ScienceDataset(
        'train1_ids_gray_only1_500',
        mode='train',
        #'valid1_ids_gray_only1_43', mode='train',
        transform=train_augment)
    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        #sampler = ConstantSampler(train_dataset,list(range(16))),
        batch_size=batch_size,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=train_collate)

    valid_dataset = ScienceDataset(
        'valid1_ids_gray_only1_43',
        mode='train',
        #'debug1_ids_gray_only1_10', mode='train',
        transform=valid_augment)
    valid_loader = DataLoader(
        valid_dataset,
        sampler=SequentialSampler(valid_dataset),
        batch_size=batch_size,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=train_collate)

    log.write('\ttrain_dataset.split = %s\n' % (train_dataset.split))
    log.write('\tvalid_dataset.split = %s\n' % (valid_dataset.split))
    log.write('\tlen(train_dataset)  = %d\n' % (len(train_dataset)))
    log.write('\tlen(valid_dataset)  = %d\n' % (len(valid_dataset)))
    log.write('\tlen(train_loader)   = %d\n' % (len(train_loader)))
    log.write('\tlen(valid_loader)   = %d\n' % (len(valid_loader)))
    log.write('\tbatch_size  = %d\n' % (batch_size))
    log.write('\titer_accum  = %d\n' % (iter_accum))
    log.write('\tbatch_size*iter_accum  = %d\n' % (batch_size * iter_accum))
    log.write('\n')

    #log.write(inspect.getsource(train_augment)+'\n')
    #log.write(inspect.getsource(valid_augment)+'\n')
    #log.write('\n')

    if 0:  #<debug>
        for inputs, truth_boxes, truth_labels, truth_instances, indices in valid_loader:

            batch_size, C, H, W = inputs.size()
            print(batch_size)

            images = inputs.cpu().numpy()
            for b in range(batch_size):
                image = (images[b].transpose((1, 2, 0)) * 255)
                image = np.clip(image.astype(np.float32) * 3, 0, 255)

                image1 = image.copy()

                truth_box = truth_boxes[b]
                truth_label = truth_labels[b]
                truth_instance = truth_instances[b]
                if truth_box is not None:
                    for box, label, instance in zip(truth_box, truth_label, truth_instance):
                        x0, y0, x1, y1 = box.astype(np.int32)
                        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 1)
                        print(label)

                        thresh = instance > 0.5
                        contour = thresh_to_inner_contour(thresh)
                        contour = contour.astype(np.float32) * 0.5

                        image1 = contour[:, :, np.newaxis] * np.array(
                            (0, 255, 0)) + (1 - contour[:, :, np.newaxis]) * image1

                    print('')

                image_show('image', image)
                image_show('image1', image1)
                cv2.waitKey(0)

    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write(' optimizer=%s\n' % str(optimizer))
    log.write(' momentum=%f\n' % optimizer.param_groups[0]['momentum'])
    log.write(' LR=%s\n\n' % str(LR))

    log.write(' images_per_epoch = %d\n\n' % len(train_dataset))
    log.write(
        ' rate    iter   epoch  num   | valid_loss                           | train_loss                           | batch_loss                           |  time    \n'
    )
    log.write(
        '------------------------------------------------------------------------------------------------------------------------------------------------------------------\n'
    )

    train_loss = np.zeros(6, np.float32)
    train_acc = 0.0
    valid_loss = np.zeros(6, np.float32)
    valid_acc = 0.0
    batch_loss = np.zeros(6, np.float32)
    batch_acc = 0.0
    rate = 0

    start = timer()
    j = 0
    i = 0

    while i < num_iters:  # loop over the dataset multiple times
        sum_train_loss = np.zeros(6, np.float32)
        sum_train_acc = 0.0
        sum = 0

        net.set_mode('train')
        optimizer.zero_grad()
        for inputs, truth_boxes, truth_labels, truth_instances, indices in train_loader:
            batch_size = len(indices)
            i = j / iter_accum + start_iter
            epoch = (i - start_iter) * batch_size * iter_accum / len(train_dataset) + start_epoch
            num_products = epoch * len(train_dataset)

            if i % iter_valid == 0:
                net.set_mode('valid')
                valid_loss, valid_acc = evaluate(net, valid_loader)
                net.set_mode('train')

                print('\r', end='', flush=True)
                log.write('%0.4f %5.1f k %6.2f %4.1f m | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %s\n' % (\
                         rate, i/1000, epoch, num_products/1000000,
                         valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3], valid_loss[4], valid_loss[5], #valid_acc,
                         train_loss[0], train_loss[1], train_loss[2], train_loss[3], train_loss[4], train_loss[5], #train_acc,
                         batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[3], batch_loss[4], batch_loss[5], #batch_acc,
                         time_to_str((timer() - start)/60)))
                time.sleep(0.01)

            #if 1:
            if i in iter_save:
                torch.save(net.state_dict(), out_dir + '/checkpoint/%08d_model.pth' % (i))
            """
                torch.save({
                    'optimizer': optimizer.state_dict(),
                    'iter'     : i,
                    'epoch'    : epoch,
                }, out_dir +'/checkpoint/%08d_optimizer.pth'%(i))

            """

            # learning rate schduler -------------
            if LR is not None:
                lr = LR.get_rate(i)
                if lr < 0: break
                adjust_learning_rate(optimizer, lr / iter_accum)
            rate = get_learning_rate(optimizer)[0] * iter_accum

            # one iteration update  -------------
            inputs = Variable(inputs).cuda()
            net(inputs, truth_boxes, truth_labels, truth_instances)
            loss = net.loss(inputs, truth_boxes, truth_labels, truth_instances)

            if 0:  #<debug>
                debug_and_draw(net, inputs, truth_boxes, truth_labels, truth_instances, mode='test')

            # masks  = (probs>0.5).float()
            # acc    = dice_loss(masks, labels)

            # accumulated update
            loss.backward()
            if j % iter_accum == 0:
                #torch.nn.utils.clip_grad_norm(net.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()

            # print statistics  ------------
            batch_acc = 0  #acc[0][0]
            batch_loss = np.array((
                loss.cpu().data.numpy()[0],
                net.rpn_cls_loss.cpu().data.numpy()[0],
                net.rpn_reg_loss.cpu().data.numpy()[0],
                net.rcnn_cls_loss.cpu().data.numpy()[0],
                net.rcnn_reg_loss.cpu().data.numpy()[0],
                net.mask_cls_loss.cpu().data.numpy()[0],
            ))
            sum_train_loss += batch_loss
            sum_train_acc += batch_acc
            sum += 1
            if i % iter_smooth == 0:
                train_loss = sum_train_loss / sum
                train_acc = sum_train_acc / sum
                sum_train_loss = np.zeros(6, np.float32)
                sum_train_acc = 0.
                sum = 0


            print('\r%0.4f %5.1f k %6.2f %4.1f m | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %s  %d,%d,%s' % (\
                         rate, i/1000, epoch, num_products/1000000,
                         valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3], valid_loss[4], valid_loss[5], #valid_acc,
                         train_loss[0], train_loss[1], train_loss[2], train_loss[3], train_loss[4], train_loss[5], #train_acc,
                         batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[3], batch_loss[4], batch_loss[5], #batch_acc,
                         time_to_str((timer() - start)/60) ,i,j, str(inputs.size())), end='',flush=True)
            j = j + 1

        pass  #-- end of one data loader --
    pass  #-- end of all iterations --

    if 1:  #save last
        torch.save(net.state_dict(), out_dir + '/checkpoint/%d_model.pth' % (i))
        """
        torch.save({
            'optimizer': optimizer.state_dict(),
            'iter': i,
            'epoch': epoch,
        }, out_dir + '/checkpoint/%d_optimizer.pth' % (i))
        """
    log.write('\n')


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    run_train()

    print('\nsucess!')

#  ffmpeg -f image2  -pattern_type glob -r 33 -i "iterations/*.png" -c:v libx264  iterations.mp4
#
#
