# import os, sys
# sys.path.append(os.path.dirname(__file__))
#
# from train import *
#
#
#
# ## overwrite functions ###
#
# def eval_augment(image,label,index):
#
#     label = label.astype(np.float32)
#     image, label = fix_crop_transform2(image, label, -1, -1, WIDTH, HEIGHT)
#     label = label.astype(np.int32)
#
#     center, delta = label_to_multi_center(label, HEADS)
#     center = center.astype(np.float32)
#
#     #---------------------------------------
#     tensor = image.transpose((2,0,1))
#     tensor = torch.from_numpy(tensor).float().div(255)
#     truth0  = torch.from_numpy(center)
#     truth1  = torch.from_numpy(delta)
#
#     return tensor, truth0, truth1, index
#
#
# #eval_augment = valid_augment
#
#
#
# def show_evaluate(tensors, labels, probs, indices, ids, wait=1, is_save=True, dir=None):
#
#     os.makedirs(dir, exist_ok=True)
#     os.makedirs(dir + '/all', exist_ok=True)
#     os.makedirs(dir + '/mask', exist_ok=True)
#
#     batch_size,C,H,W = tensors.size()
#     #print(batch_size)
#
#     images = tensors.data.cpu().numpy()
#     labels = labels.data.cpu().numpy()
#     probs  = probs.data.cpu().numpy()
#     for m in range(batch_size):
#         image = images[m].transpose((1,2,0))*255
#         image = image.astype(np.uint8)
#
#         label = labels[m]
#         prob  = probs[m]
#
#         overlay = draw_multi_center(image, prob>0.5)
#
#         label = label.sum(0)
#         prob  = np.clip(prob.sum(0),0,1)
#         label = label.reshape(H,W)[:, :, np.newaxis]*np.array([255,255,255])
#         prob  = prob. reshape(H,W)[:, :, np.newaxis]*np.array([255,255,255])
#         all   = np.hstack((image, overlay, label, prob))
#         all   = all.astype(np.uint8)
#
#         if is_save == True:
#             id = ids[indices[m]]
#             name =id.split('/')[-1]
#             cv2.imwrite(dir +'/all/%s.png'%(name),all)
#             cv2.imwrite(dir +'/mask/%s.png'%(name),prob)
#
#         # image_show('image',image)
#         # image_show('label',label)
#         # image_show('prob',prob)
#         image_show('all',all)
#         cv2.waitKey(wait)
#
#
#
# def make_train_ellipses(tensors, probs, deltas):
#
#     batch_size,C,H,W = tensors.size()
#     images  = tensors.data.cpu().numpy()
#     probs   = probs.data.cpu().numpy()
#     deltas  = deltas.data.cpu().numpy().reshape(batch_size,NUM_HEADS,6,H,W)
#
#     ellipses = []
#     for m in range(batch_size):
#         prob  = probs[m]
#         delta = deltas[m]
#
#         candidates = prob_delta_to_candidates( prob, delta, threshold = 0.3)
#         nms = non_max_suppress(candidates, min_distance_threshold=5)
#         ellipses.append(nms)
#
#     return ellipses
#
# #--------------------------------------------------------------
# def run_evaluate():
#
#     out_dir  = RESULTS_DIR + '/MultiSegmentNet-4cls-multi_center-delta-128-gray-06'
#     initial_checkpoint = \
#         out_dir + '/checkpoint/00018000_model.pth'
#
#
#
#     ## setup  ---------------------------
#     os.makedirs(out_dir +'/checkpoint', exist_ok=True)
#     os.makedirs(out_dir +'/backup', exist_ok=True)
#     backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.%s.zip'%IDENTIFIER)
#
#     log = Logger()
#     log.open(out_dir+'/log.evaluate.txt',mode='a')
#     log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
#     log.write('** some experiment setting **\n')
#     log.write('\tSEED         = %u\n' % SEED)
#     log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
#     log.write('\tout_dir      = %s\n' % out_dir)
#     log.write('\n')
#
#
#     ## net ------------------------------
#     log.write('** net setting **\n')
#     net = Net(in_shape = (3,HEIGHT,WIDTH), num_classes=NUM_CLASSES).cuda()
#
#     if initial_checkpoint is not None:
#         log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
#         net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
#
#
#     log.write('%s\n\n'%(type(net)))
#     log.write('\n')
#
#
#
#     ## dataset ----------------------------------------
#     log.write('** dataset setting **\n')
#
#     test_dataset = ScienceDataset(
#                                 #'train1_ids_gray_only_500', mode='train',
#                                 'valid1_ids_gray_only_43', mode='train',
#                                 transform = eval_augment)
#     test_loader  = DataLoader(
#                         test_dataset,
#                         sampler = SequentialSampler(test_dataset),
#                         batch_size  = 1,
#                         drop_last   = False,
#                         num_workers = 4,
#                         pin_memory  = True,
#                         collate_fn  = train_collate)
#
#
#     log.write('\ttest_dataset.split = %s\n'%(test_dataset.split))
#     log.write('\tlen(test_dataset)  = %d\n'%(len(test_dataset)))
#     log.write('\n')
#
#
#
#
#
#     ## start evaluation here! ##############################################
#     log.write('** start evaluation here! **\n')
#     net.eval()
#
#     test_num  = 0
#     test_loss = 0
#     test_acc  = 0
#     for i, (tensors, truths0, truths1, indices) in enumerate(test_loader, 0):
#
#         tensors = Variable(tensors,volatile=True).cuda()
#         truths0 = Variable(truths0).cuda()
#         truths1 = Variable(truths1).cuda()
#
#         logits, probs, deltas = data_parallel(net, tensors)
#         #loss = BCELoss2d()(logits, labels)
#         #loss = WeightedBCELoss2d()(logits, masks, weights)
#
#
#         ##save results ---------------------------------------
#         batch_size,C,H,W = tensors.size()
#         ids = test_dataset.ids
#
#         images  = tensors.data.cpu().numpy()
#         probs   = probs.data.cpu().numpy()
#         deltas  = deltas.data.cpu().numpy().reshape(batch_size,NUM_HEADS,6,H,W)
#         for m in range(batch_size):
#             original_image = original_images[m]
#             image = (images[m].transpose((1,2,0))*255).astype(np.uint8)
#             prob  = probs[m]
#             delta = deltas[m]
#
#
#             candidates = prob_delta_to_candidates( prob, delta, threshold=0.4)
#             nms = non_max_suppress(candidates, min_distance_threshold=1)
#             original_nms = nms_to_original_size( nms, image, original_image)
#             label = nms_to_label( original_nms, original_image )
#             label_overlay = (skimage.color.label2rgb(label, bg_label=0, bg_color=(0, 0, 0))*255).astype(np.uint8)
#             label_overlay = draw_label_contour(label_overlay, label, (255,255,255))
#
#
#             ##draw results ----
#             prob_overlay  = draw_multi_center(image, prob>0.4)
#
#
#             delta_overlay = image.copy()
#             candidates = candidates.astype(np.int32)
#             for candidate in candidates:
#                 cx,cy, minor_r,major_r, angle, score = candidate
#                 cv2.ellipse(delta_overlay, (cx,cy), (minor_r,major_r), angle, 0, 360, (255,0,255), 1)
#
#
#             ##draw results ---
#             nms_overlay = image.copy()
#             nms = nms.astype(np.int32)
#             for candidate in nms:
#                 cx,cy, minor_r,major_r, angle, score = candidate
#                 cv2.ellipse(nms_overlay, (cx,cy), (minor_r,major_r), angle, 0, 360, (0,255,255), 1, cv2.LINE_AA)
#
#             original_nms_overlay = original_image.copy()
#             original_nms = original_nms.astype(np.int32)
#             for candidate in original_nms:
#                 cx,cy, minor_r,major_r, angle, score = candidate
#                 cv2.ellipse(original_nms_overlay, (cx,cy), (minor_r,major_r), angle, 0, 360, (0,255,255), 1, cv2.LINE_AA)
#
#
#
#
#
#         if 1: #<debug>
#             show_evaluate(tensors, labels, probs, indices, test_dataset.ids,
#                              wait=1, is_save=True, dir=out_dir +'/eval/'+test_dataset.split )
#
#         # print statistics  ------------
#         batch_size = len(indices)
#         test_acc  += batch_size*acc[0][0]
#         test_loss += batch_size*loss.data[0]
#         test_num  += batch_size
#
#     assert(test_num == len(test_loader.sampler))
#     test_acc  = test_acc/test_num
#     test_loss = test_loss/test_num
#
#     log.write('initial_checkpoint  = %s\n'%(initial_checkpoint))
#     log.write('test_acc  = %0.5f\n'%(test_acc))
#     log.write('test_loss = %0.5f\n'%(test_loss))
#     log.write('test_num  = %d\n'%(test_num))
#     log.write('\n')
#
#
#
# ## post process here ####-------------------------------------
# def run_evaluate_post_process():
#
#     out_dir  = RESULTS_DIR + '/unet2-1cls-mask-256-gray-00'
#     split= 'valid1_ids_gray_only_43'
#
#     #------------------------------------------------------------------
#     log = Logger()
#     log.open(out_dir+'/log.evaluate.txt',mode='a')
#
#     os.makedirs(out_dir +'/eval/'+split+'/label', exist_ok=True)
#     os.makedirs(out_dir +'/eval/'+split+'/final', exist_ok=True)
#
#
#     image_files = glob.glob(out_dir +'/eval/'+split+'/mask/*.png')
#     image_files.sort()
#
#     average_precisions = []
#     for image_file in image_files:
#         #image_file = image_dir + '/0a849e0eb15faa8a6d7329c3dd66aabe9a294cccb52ed30a90c8ca99092ae732.png'
#
#         name  = image_file.split('/')[-1].replace('.png','')
#         true_label = np.load(DATA_DIR    + '/stage1_train/' + name + '/label.npy')
#         image = cv2.imread(DATA_DIR + '/stage1_train/' + name + '/images/' + name +'.png')
#         H,W = image.shape[:2]
#
#         mask = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE)
#         if ((H,W)!=mask.shape[:2]):
#             mask = cv2.resize(mask,(W,H))
#         thresh = (mask>128)
#
#
#         #image_show('image',image)
#         #image_show('mask',mask)
#         #cv2.waitKey(0)
#
#
#         #baseline labeling  -------------------------
#         label = skimage.morphology.label(thresh)
#
#         # fill hole, file small, etc ...
#         # label = filter_small(label, threshold=15)
#
#
#         average_precision, precision = compute_average_precision(label, true_label)
#         average_precisions.append(average_precision)
#
#         #save and show  -------------------------
#
#         overlay = (skimage.color.label2rgb(label, bg_label=0, bg_color=(0, 0, 0))*255).astype(np.uint8)
#         cv2.imwrite(out_dir +'/eval/'+split+'/label/' + name + '.png',overlay)
#         np.save    (out_dir +'/eval/'+split+'/label/' + name + '.npy',label)
#
#
#         overlay1 = draw_label_contour (image, label )
#         mask  = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
#         final = np.hstack((image, overlay1, overlay, mask))
#         final = final.astype(np.uint8)
#         cv2.imwrite(out_dir +'/eval/'+split+'/final/' + name + '.png',final)
#
#
#         image_show('image',image)
#         image_show('mask',mask)
#         image_show('overlay',overlay)
#         cv2.waitKey(1)
#
#     ##----------------------------------------------
#     average_precisions = np.array(average_precisions)
#     log.write('-------------\n')
#     log.write('average_precision = %0.5f\n'%average_precisions.mean())
#     log.write('\n')
#
#
#
#
#
#
#
#
#
# # main #################################################################
# if __name__ == '__main__':
#     print( '%s: calling main function ... ' % os.path.basename(__file__))
#
#
#     run_evaluate()
#     #run_evaluate_post_process()
#
#
#     print('\nsucess!')