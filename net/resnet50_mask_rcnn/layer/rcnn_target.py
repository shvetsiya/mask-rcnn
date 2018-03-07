# reference:  https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/rpn/anchor_target_layer.py
from common import *
from utility.draw import *
from net.lib.box.process import *

if __name__ == '__main__':
    from rcnn_nms import *
else:
    from .rcnn_nms import *


def add_truth_box_to_proposal(cfg, proposal, b, truth_box, truth_label, score=-1):

    #proposal i,x0,y0,x1,y1,score, label
    if len(truth_box) != 0:
        truth = np.zeros((len(truth_box), 7), np.float32)
        truth[:, 0] = b
        truth[:, 1:5] = truth_box
        truth[:, 5] = score  #1  #
        truth[:, 6] = truth_label
    else:
        truth = np.zeros((0, 7), np.float32)

    sampled_proposal = np.vstack([proposal, truth])
    return sampled_proposal


# gpu version
## see https://github.com/ruotianluo/pytorch-faster-rcnn
def make_one_rcnn_target(cfg, input, proposal, truth_box, truth_label):
    sampled_proposal = Variable(torch.FloatTensor((0, 7))).cuda()
    sampled_label = Variable(torch.LongTensor((0, 1))).cuda()
    sampled_assign = np.array((0, 1), np.int32)
    sampled_target = Variable(torch.FloatTensor((0, 4))).cuda()

    if len(truth_box) == 0 or len(proposal) == 0:
        return sampled_proposal, sampled_label, sampled_assign, sampled_target

    #filter invalid proposal ---------------
    _, height, width = input.size()
    num_proposal = len(proposal)

    valid = []
    for i in range(num_proposal):
        box = proposal[i, 1:5]
        if not (is_small_box(box, min_size=cfg.mask_train_min_size)):  #is_small_box_at_boundary
            valid.append(i)

    if len(valid) == 0:
        return sampled_proposal, sampled_label, sampled_assign, sampled_target

    proposal = proposal[valid]
    #----------------------------------------

    num_proposal = len(proposal)
    box = proposal[:, 1:5]

    overlap = cython_box_overlap(box, truth_box)
    argmax_overlap = np.argmax(overlap, 1)
    max_overlap = overlap[np.arange(num_proposal), argmax_overlap]

    fg_index = np.where(max_overlap >= cfg.rcnn_train_fg_thresh_low)[0]
    bg_index = np.where((max_overlap <  cfg.rcnn_train_bg_thresh_high) & \
                        (max_overlap >= cfg.rcnn_train_bg_thresh_low))[0]

    # sampling for class balance
    num_classes = cfg.num_classes
    num = cfg.rcnn_train_batch_size
    num_fg = int(np.round(cfg.rcnn_train_fg_fraction * cfg.rcnn_train_batch_size))

    # Small modification to the original version where we ensure a fixed number of regions are sampled
    # https://github.com/precedenceguo/mx-rcnn/commit/3853477d9155c1f340241c04de148166d146901d
    fg_length = len(fg_index)
    bg_length = len(bg_index)
    #print(fg_inds_length)

    if fg_length > 0 and bg_length > 0:
        num_fg = min(num_fg, fg_length)
        fg_index = fg_index[np.random.choice(fg_length, size=num_fg, replace=fg_length < num_fg)]
        num_bg = num - num_fg
        bg_index = bg_index[np.random.choice(bg_length, size=num_bg, replace=bg_length < num_bg)]

    elif fg_length > 0:  #no bgs
        num_fg = num
        num_bg = 0
        fg_index = fg_index[np.random.choice(fg_length, size=num_fg, replace=fg_length < num_fg)]

    elif bg_length > 0:  #no fgs
        num_fg = 0
        num_bg = num
        bg_index = bg_index[np.random.choice(bg_length, size=num_bg, replace=bg_length < num_bg)]
        num_fg_proposal = 0
    else:
        # no bgs and no fgs?
        # raise NotImplementedError
        num_fg = 0
        num_bg = num
        bg_index = np.random.choice(num_proposal, size=num_bg, replace=num_proposal < num_bg)

    assert ((num_fg + num_bg) == num)

    # selecting both fg and bg
    index = np.concatenate([fg_index, bg_index], 0)
    sampled_proposal = proposal[index]

    #label
    sampled_assign = argmax_overlap[index]
    sampled_label = truth_label[sampled_assign]
    sampled_label[num_fg:] = 0  # Clamp labels for the background to 0

    #target
    if num_fg > 0:
        target_truth_box = truth_box[sampled_assign[:num_fg]]
        target_box = sampled_proposal[:num_fg][:, 1:5]
        sampled_target = rcnn_encode(target_box, target_truth_box)

    sampled_target = Variable(torch.from_numpy(sampled_target)).cuda()
    sampled_label = Variable(torch.from_numpy(sampled_label)).long().cuda()
    sampled_proposal = Variable(torch.from_numpy(sampled_proposal)).cuda()

    return sampled_proposal, sampled_label, sampled_assign, sampled_target


def make_rcnn_target(cfg, mode, inputs, proposals, truth_boxes, truth_labels):

    #<todo> take care of don't care ground truth. Here, we only ignore them  ----
    truth_boxes = copy.deepcopy(truth_boxes)
    truth_labels = copy.deepcopy(truth_labels)
    batch_size = len(inputs)
    for b in range(batch_size):
        index = np.where(truth_labels[b] > 0)[0]
        truth_boxes[b] = truth_boxes[b][index]
        truth_labels[b] = truth_labels[b][index]
    #----------------------------------------------------------------------------

    proposals = proposals.cpu().data.numpy()
    sampled_proposals = []
    sampled_labels = []
    sampled_assigns = []
    sampled_targets = []

    batch_size = len(truth_boxes)
    for b in range(batch_size):
        input = inputs[b]
        truth_box = truth_boxes[b]
        truth_label = truth_labels[b]

        if len(truth_box) != 0:
            if len(proposals) == 0:
                proposal = np.zeros((0, 7), np.float32)
            else:
                proposal = proposals[proposals[:, 0] == b]

            proposal = add_truth_box_to_proposal(cfg, proposal, b, truth_box, truth_label)


            sampled_proposal, sampled_label, sampled_assign, sampled_target = \
                make_one_rcnn_target(cfg, input, proposal, truth_box, truth_label)

            sampled_proposals.append(sampled_proposal)
            sampled_labels.append(sampled_label)
            sampled_assigns.append(sampled_assign)
            sampled_targets.append(sampled_target)

    sampled_proposals = torch.cat(sampled_proposals, 0)
    sampled_labels = torch.cat(sampled_labels, 0)
    sampled_targets = torch.cat(sampled_targets, 0)
    sampled_assigns = np.hstack(sampled_assigns)

    return sampled_proposals, sampled_labels, sampled_assigns, sampled_targets


#-----------------------------------------------------------------------------
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    check_layer()
