import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #'3,2' #'3,2,1,0'

from common import *
from utility.file import *
from dataset.reader import *
from net.rate import *
from net.metric import *

# -------------------------------------------------------------------------------------
#WIDTH, HEIGHT = 128,128
#WIDTH, HEIGHT = 192,192
WIDTH, HEIGHT = 256, 256

from net.resnet50_mask_rcnn.draw import *
from net.resnet50_mask_rcnn.model import *

# -------------------------------------------------------------------------------------


def train_augment(image, multi_mask, meta, index):

    image, multi_mask = random_shift_scale_rotate_transform2(
        image,
        multi_mask,
        shift_limit=[0, 0],
        scale_limit=[1 / 2, 2],
        rotate_limit=[-45, 45],
        borderMode=cv2.BORDER_REFLECT_101,
        u=0.5)  #borderMode=cv2.BORDER_CONSTANT

    # overlay = multi_mask_to_color_overlay(multi_mask,color='cool')
    # overlay1 = multi_mask_to_color_overlay(multi_mask1,color='cool')
    # image_show('overlay',overlay)
    # image_show('overlay1',overlay1)
    # cv2.waitKey(0)

    image, multi_mask = random_crop_transform2(image, multi_mask, WIDTH, HEIGHT, u=0.5)
    image, multi_mask = random_horizontal_flip_transform2(image, multi_mask, 0.5)
    image, multi_mask = random_vertical_flip_transform2(image, multi_mask, 0.5)
    image, multi_mask = random_rotate90_transform2(image, multi_mask, 0.5)
    ##image,  multi_mask = fix_crop_transform2(image, multi_mask, -1,-1,WIDTH, HEIGHT)

    #---------------------------------------
    input = torch.from_numpy(image.transpose((2, 0, 1))).float().div(255)
    box, label, instance = multi_mask_to_annotation(multi_mask)

    return input, box, label, instance, meta, index


def valid_augment(image, multi_mask, meta, index):

    image, multi_mask = fix_crop_transform2(image, multi_mask, -1, -1, WIDTH, HEIGHT)

    #---------------------------------------
    input = torch.from_numpy(image.transpose((2, 0, 1))).float().div(255)
    box, label, instance = multi_mask_to_annotation(multi_mask)

    return input, box, label, instance, meta, index


def train_collate(batch):

    batch_size = len(batch)
    #for b in range(batch_size): print (batch[b][0].size())
    inputs = torch.stack([batch[b][0] for b in range(batch_size)], 0)
    boxes = [batch[b][1] for b in range(batch_size)]
    labels = [batch[b][2] for b in range(batch_size)]
    instances = [batch[b][3] for b in range(batch_size)]
    metas = [batch[b][4] for b in range(batch_size)]
    indices = [batch[b][5] for b in range(batch_size)]

    return [inputs, boxes, labels, instances, metas, indices]


### training ##############################################################
def evaluate(net, test_loader):

    test_num = 0
    test_loss = np.zeros(6, np.float32)
    test_acc = 0
    for i, (inputs, truth_boxes, truth_labels, truth_instances, metas, indices) in enumerate(
            test_loader, 0):

        with torch.no_grad():
            inputs = Variable(inputs).cuda()
            net(inputs, truth_boxes, truth_labels, truth_instances)
            loss = net.loss(inputs, truth_boxes, truth_labels, truth_instances)

        # acc    = dice_loss(masks, labels) #todo

        batch_size = len(indices)
        test_acc += 0  #batch_size*acc[0][0]
        test_loss += batch_size * np.array((
            loss.cpu().data.numpy(),
            net.rpn_cls_loss.cpu().data.numpy(),
            net.rpn_reg_loss.cpu().data.numpy(),
            net.rcnn_cls_loss.cpu().data.numpy(),
            net.rcnn_reg_loss.cpu().data.numpy(),
            net.mask_cls_loss.cpu().data.numpy(),
        ))
        test_num += batch_size

    assert (test_num == len(test_loader.sampler))
    test_acc = test_acc / test_num
    test_loss = test_loss / test_num
    return test_loss, test_acc


#--------------------------------------------------------------
def run_train():
    out_dir = RESULTS_DIR + '/mask-rcnn-50-gray500-02'
<<<<<<< HEAD
    initial_checkpoint = RESULTS_DIR + '/mask-rcnn-50-gray500-02/checkpoint/00022500_model.pth'
=======
    initial_checkpoint = RESULTS_DIR + '/mask-rcnn-50-gray500-02/checkpoint/best_model.pth'
>>>>>>> 56c15d8dddfc4e812b7c44c5498b81bec17acb35
    ##

    pretrain_file = RESULTS_DIR + '/mask-rcnn-50-gray500-02/checkpoint/00022500_model.pth'
    #None #RESULTS_DIR + '/mask-single-shot-dummy-1a/checkpoint/00028000_model.pth'
    skip = ['crop', 'mask']

    ## setup  -----------------
    os.makedirs(out_dir + '/checkpoint', exist_ok=True)
    os.makedirs(out_dir + '/train', exist_ok=True)
    #os.makedirs(out_dir + '/backup', exist_ok=True)
    #backup_project_as_zip(PROJECT_PATH, out_dir + '/backup/code.train.%s.zip' % IDENTIFIER)

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
        #with open(out_dir +'/checkpoint/configuration.pkl','rb') as pickle_file:
        #    cfg = pickle.load(pickle_file)

    if pretrain_file is not None:
        log.write('\tpretrain_file = %s\n' % pretrain_file)
        net.load_pretrain(pretrain_file, skip)

    log.write('%s\n\n' % (type(net)))
    log.write('%s\n' % (net.version))
    log.write('\n')

    ## optimiser ----------------------------------
    iter_accum = 1
    batch_size = 8

    num_iters = 1000 * 1000
    iter_smooth = 20
    iter_log = 50
    iter_valid = 100
    iter_save = [0, num_iters - 1] + list(range(0, num_iters, 500))  #1*1000

    LR = None  #LR = StepLR([ (0, 0.01),  (200, 0.001),  (300, -1)])
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=0.001 / iter_accum,
        momentum=0.9,
        weight_decay=0.0001)

    start_iter = 0
    start_epoch = 0.
    if 0:  #initial_checkpoint is not None:
        checkpoint = torch.load(initial_checkpoint.replace('_model.pth', '_optimizer.pth'))
        start_iter = checkpoint['iter']
        start_epoch = checkpoint['epoch']

        rate = get_learning_rate(optimizer)  #load all except learning rate
        optimizer.load_state_dict(checkpoint['optimizer'])
        adjust_learning_rate(optimizer, rate)

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    train_dataset = ScienceDataset(
        'train1_ids_gray2_500',
        mode='train',
        #'debug1_ids_gray_only_10', mode='train',
        #'disk0_ids_dummy_9', mode='train', #12
        #'train1_ids_purple_only1_101', mode='train', #12
        #'merge1_1', mode='train',
        transform=train_augment)

    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=train_collate)

    valid_dataset = ScienceDataset(
        'valid1_ids_gray2_43',
        mode='train',
        #'debug1_ids_gray_only_10', mode='train',
        #'disk0_ids_dummy_9', mode='train',
        #'train1_ids_purple_only1_101', mode='train', #12
        #'merge1_1', mode='train',
        transform=valid_augment)

    valid_loader = DataLoader(
        valid_dataset,
        sampler=SequentialSampler(valid_dataset),
        batch_size=batch_size,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=train_collate)

    log.write('\tWIDTH, HEIGHT = %d, %d\n' % (WIDTH, HEIGHT))
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

    #<debug>========================================================================================
    if 0:
        for inputs, truth_boxes, truth_labels, truth_instances, metas, indices in valid_loader:

            batch_size, C, H, W = inputs.size()
            print('batch_size=%d' % batch_size)

            images = inputs.cpu().numpy()
            for b in range(batch_size):
                image = (images[b].transpose((1, 2, 0)) * 255)
                image = np.clip(image.astype(np.float32) * 2, 0, 255)

                contour_overlay = image.copy()
                box_overlay = image.copy()

                truth_box = truth_boxes[b]
                truth_label = truth_labels[b]
                truth_instance = truth_instances[b]
                for box, label, instance in zip(truth_box, truth_label, truth_instance):
                    print('label=%d' % label)

                    x0, y0, x1, y1 = box.astype(np.int32)
                    cv2.rectangle(box_overlay, (x0, y0), (x1, y1), (0, 0, 255), 1)

                    mask = instance > 0.5
                    contour = mask_to_inner_contour(mask)
                    contour_overlay[contour] = [0, 255, 0]

                image_show('contour_overlay', contour_overlay)
                image_show('box_overlay', box_overlay)
                cv2.waitKey(0)
    #<debug>========================================================================================

    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write(' optimizer=%s\n' % str(optimizer))
    log.write(' momentum=%f\n' % optimizer.param_groups[0]['momentum'])
    log.write(' LR=%s\n\n' % str(LR))

    log.write(' images_per_epoch = %d\n\n' % len(train_dataset))
    log.write(
        ' rate    iter   epoch  num   | valid_loss                          | train_loss               | batch_loss               |  time          \n'
    )
    log.write(
        '-------------------------------------------------------------------------------------------------------------------------------\n'
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

    last_saved_model_filepath = None

    while i < num_iters:  # loop over the dataset multiple times
        sum_train_loss = np.zeros(6, np.float32)
        sum_train_acc = 0.0
        sum = 0

        net.set_mode('train')
        optimizer.zero_grad()
        for inputs, truth_boxes, truth_labels, truth_instances, metas, indices in train_loader:
            if all(len(b) == 0 for b in truth_boxes): continue

            batch_size = len(indices)
            i = j / iter_accum + start_iter
            epoch = (i - start_iter) * batch_size * iter_accum / len(train_dataset) + start_epoch
            num_products = epoch * len(train_dataset)

            if i % iter_valid == 0:
                net.set_mode('valid')
                valid_loss, valid_acc = evaluate(net, valid_loader)
                net.set_mode('train')

                print('\r', end='', flush=True)
                log.write('%0.4f %5.1f k %6.1f %4.1f m | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %s\n' % (\
                         rate, i/1000, epoch, num_products/1000000,
                         valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3], valid_loss[4], valid_loss[5],#valid_acc,
                         train_loss[0], train_loss[1], train_loss[2], train_loss[3], train_loss[4], train_loss[5],#train_acc,
                         batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[3], batch_loss[4], batch_loss[5],#batch_acc,
                         time_to_str((timer() - start)/60)))
                time.sleep(0.01)

            #if 1:
            if i in iter_save:
                torch.save(net.state_dict(), out_dir + '/checkpoint/%08d_model.pth' % (i))
                """
                torch.save({
                    'optimizer': optimizer.state_dict(),
                    'iter': i,
                    'epoch': epoch,
                }, out_dir + '/checkpoint/%08d_optimizer.pth' % (i))
                """
                with open(out_dir + '/checkpoint/configuration.pkl', 'wb') as pickle_file:
                    pickle.dump(cfg, pickle_file, pickle.HIGHEST_PROTOCOL)

            # learning rate schduler -------------
            if LR is not None:
                lr = LR.get_rate(i)
                if lr < 0: break
                adjust_learning_rate(optimizer, lr / iter_accum)
            rate = get_learning_rate(optimizer) * iter_accum

            # one iteration update  -------------
            inputs = Variable(inputs).cuda()
            net(inputs, truth_boxes, truth_labels, truth_instances)
            loss = net.loss(inputs, truth_boxes, truth_labels, truth_instances)

            # accumulated update
            loss.backward()
            if j % iter_accum == 0:
                #torch.nn.utils.clip_grad_norm(net.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()

            # print statistics  ------------
            batch_acc = 0  #acc[0][0]
            batch_loss = np.array((
                loss.cpu().data.numpy(),
                net.rpn_cls_loss.cpu().data.numpy(),
                net.rpn_reg_loss.cpu().data.numpy(),
                net.rcnn_cls_loss.cpu().data.numpy(),
                net.rcnn_reg_loss.cpu().data.numpy(),
                net.mask_cls_loss.cpu().data.numpy(),
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

            print('\r%0.4f %5.1f k %6.1f %4.1f m | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %s  %d,%d,%s' % (\

                         rate, i/1000, epoch, num_products/1000000,
                         valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3], valid_loss[4], valid_loss[5],#valid_acc,
                         train_loss[0], train_loss[1], train_loss[2], train_loss[3], train_loss[4], train_loss[5],#train_acc,
                         batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[3], batch_loss[4], batch_loss[5],#batch_acc,
                         time_to_str((timer() - start)/60) ,i,j, ''), end='',flush=True)#str(inputs.size()))
            j = j + 1

            #<debug> ===================================================================
            if 0:
                #if i%10==0:

                net.set_mode('test')
                with torch.no_grad():
                    net(inputs, truth_boxes, truth_labels, truth_instances)

                batch_size, C, H, W = inputs.size()
                images = inputs.data.cpu().numpy()
                window = net.rpn_window
                rpn_logits_flat = net.rpn_logits_flat.data.cpu().numpy()
                rpn_deltas_flat = net.rpn_deltas_flat.data.cpu().numpy()
                rpn_proposals = net.rpn_proposals.data.cpu().numpy()

                rcnn_logits = net.rcnn_logits.data.cpu().numpy()
                rcnn_deltas = net.rcnn_deltas.data.cpu().numpy()
                rcnn_proposals = net.rcnn_proposals.data.cpu().numpy()

                detections = net.detections.data.cpu().numpy()
                masks = net.masks

                #print('train',batch_size)
                for b in range(batch_size):

                    image = (images[b].transpose((1, 2, 0)) * 255)
                    image = image.astype(np.uint8)
                    #image = np.clip(image.astype(np.float32)*2,0,255).astype(np.uint8)  #improve contrast

                    truth_box = truth_boxes[b]
                    truth_label = truth_labels[b]
                    truth_instance = truth_instances[b]
                    truth_mask = instance_to_multi_mask(truth_instance)

                    rpn_logit_flat = rpn_logits_flat[b]
                    rpn_delta_flat = rpn_deltas_flat[b]
                    rpn_prob_flat = np_softmax(rpn_logit_flat)

                    rpn_proposal = np.zeros((0, 7), np.float32)
                    if len(rpn_proposals) > 0:
                        index = np.where(rpn_proposals[:, 0] == b)[0]
                        rpn_proposal = rpn_proposals[index]

                    rcnn_proposal = np.zeros((0, 7), np.float32)
                    if len(rcnn_proposals) > 0:
                        index = np.where(rcnn_proposals[:, 0] == b)[0]
                        rcnn_logit = rcnn_logits[index]
                        rcnn_delta = rcnn_deltas[index]
                        rcnn_prob = np_softmax(rcnn_logit)
                        rcnn_proposal = rcnn_proposals[index]

                    mask = masks[b]

                    #box = proposal[:,1:5]
                    #mask = masks[b]

                    ## draw --------------------------------------------------------------------------
                    #contour_overlay = multi_mask_to_contour_overlay(truth_mask, image, [255,255,0] )
                    #color_overlay   = multi_mask_to_color_overlay(mask)

                    #all1 = draw_multi_rpn_prob(cfg, image, rpn_prob_flat)
                    #all2 = draw_multi_rpn_delta(cfg, overlay_contour, rpn_prob_flat, rpn_delta_flat, window,[0,0,255])
                    #all3 = draw_multi_rpn_proposal(cfg, image, proposal)
                    #all4 = draw_truth_box(cfg, image, truth_box, truth_label)

                    all5 = draw_multi_proposal_metric(cfg, image, rpn_proposal, truth_box,
                                                      truth_label, [0, 255, 255], [255, 0, 255],
                                                      [255, 255, 0])
                    all6 = draw_multi_proposal_metric(cfg, image, rcnn_proposal, truth_box,
                                                      truth_label, [0, 255, 255], [255, 0, 255],
                                                      [255, 255, 0])
                    all7 = draw_mask_metric(cfg, image, mask, truth_box, truth_label,
                                            truth_instance)

                    # image_show('color_overlay',color_overlay,1)
                    # image_show('rpn_prob',all1,1)
                    # image_show('rpn_prob',all1,1)
                    # image_show('rpn_delta',all2,1)
                    # image_show('rpn_proposal',all3,1)
                    # image_show('truth_box',all4,1)
                    # image_show('rpn_precision',all5,1)

                    # imshow is here
                    #image_show('rpn_precision', all5, 1)
                    #image_show('rcnn_precision', all6, 1)
                    #image_show('mask_precision', all7, 1)

                    # summary = np.vstack([
                    #     all5,
                    #     np.hstack([
                    #         all1,
                    #         np.vstack( [all2, np.zeros((H,2*W,3),np.uint8)])
                    #     ])
                    # ])
                    # draw_shadow_text(summary, 'iter=%08d'%i,  (5,3*HEIGHT-15),0.5, (255,255,255), 1)
                    # image_show('summary',summary,1)

                    name = train_dataset.ids[indices[b]].split('/')[-1]
                    #cv2.imwrite(out_dir +'/train/%s.png'%name,summary)
                    #cv2.imwrite(out_dir +'/train/%05d.png'%b,summary)

                    cv2.imwrite(out_dir + '/train/%05d.rpn_precision.png' % b, all5)
                    cv2.imwrite(out_dir + '/train/%05d.rcnn_precision..png' % b, all6)
                    cv2.waitKey(1)
                    pass

                net.set_mode('train')
            #<debug> ===================================================================

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
