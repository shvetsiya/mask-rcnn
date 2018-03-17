from typing import List
import torch
from torch.autograd import Variable

from common import *
from net.lib.roi_align_pool_tf.module import RoIAlign as Crop

if __name__ == '__main__':
    from configuration import Configuration
    from layer.rpn_multi_nms import *
    from layer.rpn_multi_target import *
    from layer.rpn_multi_loss import *
    from layer.rcnn_nms import *
    from layer.rcnn_target import *
    from layer.rcnn_loss import *
    from layer.mask_nms import *
    from layer.mask_target import *
    from layer.mask_loss import *

else:
    from .configuration import Configuration
    from .layer.rpn_multi_nms import *
    from .layer.rpn_multi_target import *
    from .layer.rpn_multi_loss import *
    from .layer.rcnn_nms import *
    from .layer.rcnn_target import *
    from .layer.rcnn_loss import *
    from .layer.mask_nms import *
    from .layer.mask_target import *
    from .layer.mask_loss import *

#############  resent50 pyramid feature net ##############################################################################


## P layers ## ---------------------------
class LateralBlock(nn.Module):

    def __init__(self, c_planes, p_planes, out_planes):
        super(LateralBlock, self).__init__()
        self.lateral = nn.Conv2d(c_planes, p_planes, kernel_size=1, padding=0, stride=1)
        self.top = nn.Conv2d(p_planes, out_planes, kernel_size=3, padding=1, stride=1)

    def forward(self, c, p):
        _, _, H, W = c.size()
        c = self.lateral(c)
        p = F.upsample(p, scale_factor=2, mode='nearest')
        p = p[:, :, :H, :W] + c
        p = self.top(p)

        return p


## C layers ## ---------------------------
class BottleneckBlock(nn.Module):

    def __init__(self, in_planes, planes, out_planes, is_downsample=False, stride=1):
        super(BottleneckBlock, self).__init__()
        self.is_downsample = is_downsample

        self.bn1 = nn.BatchNorm2d(in_planes, eps=2e-5)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=2e-5)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, eps=2e-5)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, padding=0, stride=1, bias=False)

        if is_downsample:
            self.downsample = nn.Conv2d(
                in_planes, out_planes, kernel_size=1, padding=0, stride=stride, bias=False)

    def forward(self, x):

        x = F.relu(self.bn1(x), inplace=True)
        z = self.conv1(x)
        z = F.relu(self.bn2(z), inplace=True)
        z = self.conv2(z)
        z = F.relu(self.bn3(z), inplace=True)
        z = self.conv3(z)

        if self.is_downsample:
            z += self.downsample(x)
        else:
            z += x

        return z


def make_layer_c0(in_planes, out_planes):
    layers = [
        nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    ]
    return nn.Sequential(*layers)


def make_layer_c(in_planes, planes, out_planes, num_blocks, stride):
    layers = []
    layers.append(BottleneckBlock(in_planes, planes, out_planes, is_downsample=True, stride=stride))
    for i in range(1, num_blocks):
        layers.append(BottleneckBlock(out_planes, planes, out_planes))

    return nn.Sequential(*layers)


class FeatureNet(nn.Module):

    def __init__(self, cfg: Configuration, in_channels, out_channels=256):
        super(FeatureNet, self).__init__()
        self.cfg = cfg

        # bottom-top
        self.layer_c0 = make_layer_c0(in_channels, 64)

        self.layer_c1 = make_layer_c(64, 64, 256, num_blocks=3, stride=1)  #out =  64*4 =  256
        self.layer_c2 = make_layer_c(256, 128, 512, num_blocks=4, stride=2)  #out = 128*4 =  512
        self.layer_c3 = make_layer_c(512, 256, 1024, num_blocks=6, stride=2)  #out = 256*4 = 1024
        self.layer_c4 = make_layer_c(1024, 512, 2048, num_blocks=3, stride=2)  #out = 512*4 = 2048

        # top-down
        self.layer_p4 = nn.Conv2d(2048, out_channels, kernel_size=1, stride=1, padding=0)
        self.layer_p3 = LateralBlock(1024, out_channels, out_channels)
        self.layer_p2 = LateralBlock(512, out_channels, out_channels)
        self.layer_p1 = LateralBlock(256, out_channels, out_channels)

    def forward(self, x):
        #pass                        #; print('input ',   x.size())
        c0 = self.layer_c0(x)  #; print('layer_c0 ',c0.size())
        #
        c1 = self.layer_c1(c0)  #; print('layer_c1 ',c1.size())
        c2 = self.layer_c2(c1)  #; print('layer_c2 ',c2.size())
        c3 = self.layer_c3(c2)  #; print('layer_c3 ',c3.size())
        c4 = self.layer_c4(c3)  #; print('layer_c4 ',c4.size())

        p4 = self.layer_p4(c4)  #; print('layer_p4 ',p4.size())
        p3 = self.layer_p3(c3, p4)  #; print('layer_p3 ',p3.size())
        p2 = self.layer_p2(c2, p3)  #; print('layer_p2 ',p2.size())
        p1 = self.layer_p1(c1, p2)  #; print('layer_p1 ',p1.size())

        features = [p1, p2, p3, p4]
        assert (len(self.cfg.rpn_scales) == len(features))

        return features


############# various head ##############################################################################################


class RpnMultiHead(nn.Module):

    def __init__(self, cfg: Configuration, in_channels):
        super(RpnMultiHead, self).__init__()

        self.num_classes = cfg.num_classes
        self.num_scales = len(cfg.rpn_scales)
        self.num_bases = [len(b) for b in cfg.rpn_base_apsect_ratios]

        self.convs = nn.ModuleList()
        self.logits = nn.ModuleList()
        self.deltas = nn.ModuleList()
        for l in range(self.num_scales):
            channels = in_channels * 2
            self.convs.append(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1))
            self.logits.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels, self.num_bases[l] * self.num_classes, kernel_size=3, padding=1),))
            self.deltas.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        self.num_bases[l] * self.num_classes * 4,
                        kernel_size=3,
                        padding=1),))

    def forward(self, fs):
        batch_size = len(fs[0])

        logits_flat = []
        probs_flat = []
        deltas_flat = []
        for l in range(self.num_scales):  # apply multibox head to feature maps
            f = fs[l]
            f = F.relu(self.convs[l](f))

            f = F.dropout(f, p=0.5, training=self.training)
            logit = self.logits[l](f)
            delta = self.deltas[l](f)

            logit_flat = logit.permute(0, 2, 3, 1).contiguous().view(batch_size, -1,
                                                                     self.num_classes)
            delta_flat = delta.permute(0, 2, 3, 1).contiguous().view(batch_size, -1,
                                                                     self.num_classes, 4)
            logits_flat.append(logit_flat)
            deltas_flat.append(delta_flat)

        logits_flat = torch.cat(logits_flat, 1)
        deltas_flat = torch.cat(deltas_flat, 1)

        return logits_flat, deltas_flat


# https://qiita.com/yu4u/items/5cbe9db166a5d72f9eb8


class CropRoi(nn.Module):

    def __init__(self, cfg, crop_size):
        super(CropRoi, self).__init__()
        self.num_scales = len(cfg.rpn_scales)
        self.crop_size = crop_size
        self.sizes = cfg.rpn_base_sizes
        self.scales = cfg.rpn_scales

        self.crops = nn.ModuleList()
        for l in range(self.num_scales):
            self.crops.append(Crop(self.crop_size, self.crop_size, 1 / self.scales[l]))

    def forward(self, fs, proposals):
        num_proposals = len(proposals)

        ## this is  complicated. we need to decide for a given roi, which of the p0,p1, ..p3 layers to pool from
        boxes = proposals.detach().data[:, 1:5]
        sizes = boxes[:, 2:] - boxes[:, :2]
        sizes = torch.sqrt(sizes[:, 0] * sizes[:, 1])
        distances = torch.abs(sizes.view(num_proposals,1).expand(num_proposals,4) \
                              - torch.from_numpy(np.array(self.sizes,np.float32)).cuda())
        min_distances, min_index = distances.min(1)

        rois = proposals.detach().data[:, 0:5]
        rois = Variable(rois)

        crops = []
        indices = []
        for l in range(self.num_scales):
            index = (min_index == l).nonzero()

            if len(index) > 0:
                crop = self.crops[l](fs[l], rois[index].view(-1, 5))
                crops.append(crop)
                indices.append(index)

        crops = torch.cat(crops, 0)
        indices = torch.cat(indices, 0).view(-1)
        crops = crops[torch.sort(indices)[1]]
        #crops = torch.index_select(crops,0,index)

        return crops


class RcnnHead(nn.Module):

    def __init__(self, cfg, in_channels):
        super(RcnnHead, self).__init__()
        self.num_classes = cfg.num_classes
        self.crop_size = cfg.rcnn_crop_size

        self.fc1 = nn.Linear(in_channels * self.crop_size * self.crop_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.logit = nn.Linear(1024, self.num_classes)
        self.delta = nn.Linear(1024, self.num_classes * 4)

    def forward(self, crops):

        x = crops.view(crops.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        logits = self.logit(x)
        deltas = self.delta(x)

        return logits, deltas


# class CropRoi(nn.Module):
#     def __init__(self, cfg, in_channels, out_channels ):
#         super(CropRoi, self).__init__()
#         self.num_scales = len(cfg.rpn_scales)
#         self.scales     = cfg.rpn_scales
#         self.crop_size  = cfg.crop_size
#
#         self.convs = nn.ModuleList()
#         self.crops = nn.ModuleList()
#         for l in range(self.num_scales):
#             self.convs.append(
#                 nn.Conv2d( in_channels, out_channels//self.num_scales, kernel_size=1, padding=0, bias=False),
#             )
#             self.crops.append(
#                 Crop(self.crop_size, self.crop_size, 1/self.scales[l]),
#             )
#
#
#     def forward(self, fs, proposals):
#         rois = proposals[:,0:5]
#         crops=[]
#         for l in range(self.num_scales):
#             c = self.convs[l](fs[l])
#             c = self.crops[l](c,rois)
#             crops.append(c)
#         crops = torch.cat(crops,1)
#
#         return crops


class MaskHead(nn.Module):

    def __init__(self, cfg, in_channels):
        super(MaskHead, self).__init__()
        self.num_classes = cfg.num_classes

        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.up = nn.ConvTranspose2d(256, 256, kernel_size=4, padding=1, stride=2, bias=False)
        self.logit = nn.Conv2d(256, self.num_classes, kernel_size=1, padding=0, stride=1)

    def forward(self, crops):
        x = F.relu(self.bn1(self.conv1(crops)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = F.relu(self.bn4(self.conv4(x)), inplace=True)
        x = self.up(x)
        logits = self.logit(x)

        return logits


############# mask rcnn net ##############################################################################


class MaskRcnnNet(nn.Module):

    class Result(object):
        """A class which represents output of MaskRcnnNet.
        """

        def __init__(self, multi_mask: np.array, bounding_boxes: List[BoundingBox]):
            """Inits result with multi_mask and corresponding bounding boxes.

            Args:
                multi_mask: Int array of input image size. 0 means background, k (k > 0) - k'th
                    object.
                bounding_boxes: A list of BoundingBox, which corresponds to the multi_mask.
            """
            self.multi_mask = multi_mask
            self.bounding_boxes = bounding_boxes

    def __init__(self, cfg: Configuration):
        super(MaskRcnnNet, self).__init__()
        self.version = 'net version \'mask-rcnn-resnet50-fpn\''
        self.cfg = cfg
        self.mode = 'train'

        feature_channels = 128
        crop_channels = feature_channels
        self.feature_net = FeatureNet(cfg, 3, feature_channels)
        self.rpn_head = RpnMultiHead(cfg, feature_channels)
        self.rcnn_crop = CropRoi(cfg, cfg.rcnn_crop_size)
        self.rcnn_head = RcnnHead(cfg, crop_channels)
        self.mask_crop = CropRoi(cfg, cfg.mask_crop_size)
        self.mask_head = MaskHead(cfg, crop_channels)

    def forward(self, inputs, truth_boxes=None, truth_labels=None, truth_instances=None):
        cfg = self.cfg
        mode = self.mode
        batch_size = len(inputs)

        # Features
        features = data_parallel(self.feature_net, inputs)

        # RPN proposals
        self._rpn_logits_flat, self._rpn_deltas_flat = data_parallel(self.rpn_head, features)
        rpn_window = make_rpn_windows(cfg, features)
        rpn_proposals = rpn_nms(cfg, mode, inputs, rpn_window, self._rpn_logits_flat,
                                self._rpn_deltas_flat)

        if mode in ['train', 'valid']:
            self._rpn_labels, _, self._rpn_label_weights, self._rpn_targets, self._rpn_target_weights = make_rpn_target(
                cfg, mode, inputs, rpn_window, truth_boxes, truth_labels)

            rpn_proposals, self._rcnn_labels, _, self._rcnn_targets  = \
                make_rcnn_target(cfg, mode, inputs, rpn_proposals, truth_boxes, truth_labels )

        # RCNN proposals
        rcnn_proposals = rpn_proposals
        if len(rpn_proposals) > 0:
            rcnn_crops = self.rcnn_crop(features, rpn_proposals)
            self._rcnn_logits, self._rcnn_deltas = data_parallel(self.rcnn_head, rcnn_crops)
            rcnn_proposals = rcnn_nms(cfg, mode, inputs, rpn_proposals, self._rcnn_logits,
                                      self._rcnn_deltas)

        if mode in ['train', 'valid']:
            rcnn_proposals, self._mask_labels, _, self._mask_instances, = make_mask_target(
                cfg, mode, inputs, rcnn_proposals, truth_boxes, truth_labels, truth_instances)

        # Segmentation
        self._detections = rcnn_proposals

        if len(self._detections) > 0:
            # ROI crop
            mask_crops = self.mask_crop(features, self._detections)

            # Mask head
            self._mask_logits = data_parallel(self.mask_head, mask_crops)

        self.results = self._construct_results(cfg, inputs, self._detections, self._mask_logits)

    def _construct_results(self, cfg: Configuration, inputs: Variable, _detections: np.array,
                           _mask_logits: np.array) -> List[Result]:
        masks, bounding_boxes = masks_nms_for_batch(cfg, inputs, _detections, _mask_logits)
        results = []
        for index_in_batch in range(len(inputs)):
            results.append(self.Result(masks[index_in_batch], bounding_boxes[index_in_batch]))
        return results

    def loss(self, inputs, truth_boxes, truth_labels, truth_instances):
        cfg = self.cfg

        self.rpn_cls_loss, self.rpn_reg_loss = rpn_loss(
            self._rpn_logits_flat, self._rpn_deltas_flat, self._rpn_labels, self._rpn_label_weights,
            self._rpn_targets, self._rpn_target_weights)

        self.rcnn_cls_loss, self.rcnn_reg_loss = rcnn_loss(self._rcnn_logits, self._rcnn_deltas,
                                                           self._rcnn_labels, self._rcnn_targets)

        ## self.mask_cls_loss = Variable(torch.cuda.FloatTensor(1).zero_()).sum()
        # TODO(alexander): self._mask_logits can be not updated at `forward` step.
        self.mask_cls_loss = mask_loss(self._mask_logits, self._mask_labels, self._mask_instances)

        self.total_loss = self.rpn_cls_loss + self.rpn_reg_loss \
                          + self.rcnn_cls_loss +  self.rcnn_reg_loss \
                          + self.mask_cls_loss

        return self.total_loss

    #<todo> freeze bn for imagenet pretrain
    def set_mode(self, mode):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError

    def load_pretrain(self, pretrain_file, skip=[]):
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()

        keys = list(state_dict.keys())
        for key in keys:
            if any(s in key for s in skip): continue
            state_dict[key] = pretrain_state_dict[key]

        self.load_state_dict(state_dict)
        #raise NotImplementedError


# check #################################################################
def run_check_feature_net():

    batch_size = 4
    C, H, W = 3, 256, 256
    feature_channels = 128

    x = torch.randn(batch_size, C, H, W)
    inputs = Variable(x).cuda()

    cfg = Configuration()
    feature_net = FeatureNet(cfg, C, feature_channels).cuda()

    ps = feature_net(inputs)

    print('')
    num_heads = len(ps)
    for i in range(num_heads):
        p = ps[i]
        print(i, p.size())


def run_check_multi_rpn_head():

    batch_size = 8
    in_channels = 128
    H, W = 256, 256
    num_scales = 4
    feature_heights = [int(H // 2**l) for l in range(num_scales)]
    feature_widths = [int(W // 2**l) for l in range(num_scales)]

    fs = []
    for h, w in zip(feature_heights, feature_widths):
        f = np.random.uniform(-1, 1, size=(batch_size, in_channels, h, w)).astype(np.float32)
        f = Variable(torch.from_numpy(f)).cuda()
        fs.append(f)

    cfg = Configuration()
    rpn_head = RpnMultiHead(cfg, in_channels).cuda()
    logits_flat, deltas_flat = rpn_head(fs)

    print('logits_flat ', logits_flat.size())
    print('deltas_flat ', deltas_flat.size())
    print('')


def run_check_crop_head():

    #feature maps
    batch_size = 4
    in_channels = 128
    out_channels = 256
    H, W = 256, 256
    num_scales = 4
    feature_heights = [int(H // 2**l) for l in range(num_scales)]
    feature_widths = [int(W // 2**l) for l in range(num_scales)]

    fs = []
    for h, w in zip(feature_heights, feature_widths):
        f = np.random.uniform(-1, 1, size=(batch_size, in_channels, h, w)).astype(np.float32)
        f = Variable(torch.from_numpy(f)).cuda()
        fs.append(f)

    #proposal i,x0,y0,x1,y1,score, label
    proposals = []
    for b in range(batch_size):

        num_proposals = 4
        xs = np.random.randint(0, 64, num_proposals)
        ys = np.random.randint(0, 64, num_proposals)
        sizes = np.random.randint(8, 64, num_proposals)
        scores = np.random.uniform(0, 1, num_proposals)

        proposal = np.zeros((num_proposals, 7), np.float32)
        proposal[:, 0] = b
        proposal[:, 1] = xs
        proposal[:, 2] = ys
        proposal[:, 3] = xs + sizes
        proposal[:, 4] = ys + sizes
        proposal[:, 5] = scores
        proposal[:, 6] = 1
        proposals.append(proposal)

    proposals = np.vstack(proposals)
    proposals = Variable(torch.from_numpy(proposals)).cuda()

    #--------------------------------------
    cfg = Configuration()
    crop_net = CropRoi(cfg).cuda()
    crops = crop_net(fs, proposals)

    print('crops', crops.size())
    print('')
    #exit(0)

    crops = crops.data.cpu().numpy()
    proposals = proposals.data.cpu().numpy()

    #for m in range(num_proposals):
    for m in range(8):
        crop = crops[m]
        proposal = proposals[m]

        i, x0, y0, x1, y1, score, label = proposal

        print('i=%d, x0=%3d, y0=%3d, x1=%3d, y1=%3d, score=%0.2f' % (i, x0, y0, x1, y1, score))
        print(crop[0, 0, :5])
        print('')


def run_check_rcnn_head():

    num_rois = 100
    in_channels = 256
    crop_size = 14

    crops = np.random.uniform(
        -1, 1, size=(num_rois, in_channels, crop_size, crop_size)).astype(np.float32)
    crops = Variable(torch.from_numpy(crops)).cuda()

    cfg = Configuration()
    assert (crop_size == cfg.rcnn_crop_size)

    rcnn_head = RcnnHead(cfg, in_channels).cuda()
    logits, deltas = rcnn_head(crops)

    print('logits ', logits.size())
    print('deltas ', deltas.size())
    print('')


def run_check_mask_head():

    num_rois = 100
    in_channels = 256
    crop_size = 14

    crops = np.random.uniform(
        -1, 1, size=(num_rois, in_channels, crop_size, crop_size)).astype(np.float32)
    crops = Variable(torch.from_numpy(crops)).cuda()

    cfg = Configuration()
    assert (crop_size == cfg.crop_size)

    mask_head = MaskHead(cfg, in_channels).cuda()
    logits = mask_head(crops)

    print('logits ', logits.size())
    print('')


##-----------------------------------
def run_check_mask_net():

    batch_size, C, H, W = 1, 3, 128, 128
    feature_channels = 64
    inputs = np.random.uniform(-1, 1, size=(batch_size, C, H, W)).astype(np.float32)
    inputs = Variable(torch.from_numpy(inputs)).cuda()

    cfg = Configuration()
    mask_net = MaskSingleShotNet(cfg).cuda()

    mask_net.set_mode('eval')
    mask_net(inputs)

    print('rpn_logits_flat ', mask_net._rpn_logits_flat.size())
    print('_rpn_deltas_flat ', mask_net._rpn_deltas_flat.size())
    print('')


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    run_check_feature_net()
    # run_check_multi_rpn_head()
    # run_check_crop_head()
    # run_check_rcnn_head()
    # run_check_mask_head()

    #run_check_mask_net()
