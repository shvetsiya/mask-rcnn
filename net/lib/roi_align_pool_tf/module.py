import sys, os
sys.path.append(os.path.dirname(__file__))

import torch
from torch.nn.modules.module import Module
from function import CropAndResizeFunction


class CropAndResize(Module):
    """
    Crop and resize ported from tensorflow
    See more details on https://www.tensorflow.org/api_docs/python/tf/image/crop_and_resize
    """

    def __init__(self, crop_height, crop_width, extrapolation_value=0):
        super(CropAndResize, self).__init__()

        self.crop_height = crop_height
        self.crop_width = crop_width
        self.extrapolation_value = extrapolation_value

    def forward(self, image, boxes, box_ind):
        return CropAndResizeFunction(self.crop_height, self.crop_width,
                                     self.extrapolation_value)(image, boxes, box_ind)


# See more details on
#     https://github.com/ppwwyyxx/tensorpack/blob/6d5ba6a970710eaaa14b89d24aace179eb8ee1af/examples/FasterRCNN/model.py#L301
class RoIAlign(Module):

    def __init__(self, crop_height, crop_width, spatial_scale):
        super(RoIAlign, self).__init__()

        self.crop_height = crop_height
        self.crop_width = crop_width
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):

        #need to normalised (x0,y0,x1,y1) to [0,1]
        height, width = features.size()[2:4]
        ids, x0, y0, x1, y1 = torch.split(rois, 1, dim=1)
        ids = ids.int()
        x0 = x0 * self.spatial_scale
        y0 = y0 * self.spatial_scale
        x1 = x1 * self.spatial_scale
        y1 = y1 * self.spatial_scale

        if 0:
            #https://github.com/ppwwyyxx/tensorpack/issues/542
            #https://github.com/ppwwyyxx/tensorpack/blob/6d5ba6a970710eaaa14b89d24aace179eb8ee1af/examples/FasterRCNN/model.py#L316
            #-0.5 because bilinear sample assumes floating point coordinate (0.0, 0.0) is the same as pixel value (0, 0))
            scale_x = (x1 - x0) / (self.crop_width)
            scale_y = (y1 - y0) / (self.crop_height)
            nx0 = (x0 - 0.5 + scale_x / 2)
            ny0 = (y0 - 0.5 + scale_y / 2)
            nw = scale_x * self.crop_width
            nh = scale_y * self.crop_height
            nx1 = nx0 + nw
            ny1 = ny0 + nh
            nx0 = nx0 / (width - 1)
            ny0 = ny0 / (height - 1)
            nx1 = nx1 / (width - 1)
            ny1 = ny1 / (height - 1)
            boxes = torch.cat((ny0, nx0, ny1, nx1), 1)

        if 1:
            x0 = x0 / (width - 1)
            y0 = y0 / (height - 1)
            x1 = x1 / (width - 1)
            y1 = y1 / (height - 1)
            boxes = torch.cat((y0, x0, y1, x1), 1)

        boxes = boxes.detach().contiguous()
        ids = ids.detach()
        return CropAndResizeFunction(
            self.crop_height, self.crop_width, extrapolation_value=0)(features, boxes, ids)
