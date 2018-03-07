from common import *

from dataset.transform import *
from dataset.sampler import *
from utility.file import *
from utility.draw import *

from net.lib.box.process import *

#data reader  ----------------------------------------------------------------
MIN_SIZE =  6
MAX_SIZE =  128  #np.inf
IGNORE_BOUNDARY = -1
IGNORE_SMALL    = -2
IGNORE_BIG      = -3

class ScienceDataset(Dataset):

    def __init__(self, split, transform=None, mode='train'):
        super(ScienceDataset, self).__init__()
        start = timer()

        self.split = split
        self.transform = transform
        self.mode = mode

        #read split
        ids = read_list_from_file(DATA_DIR + '/split/' + split, comment='#')


        #save
        self.ids = ids

        #print
        print('\ttime = %0.2f min'%((timer() - start) / 60))
        print('\tnum_ids = %d'%(len(self.ids)))
        print('')


    def __getitem__(self, index):
        id = self.ids[index]
        name   = id.split('/')[-1]
        folder = id.split('/')[0]
        image = cv2.imread(DATA_DIR + '/image/%s/images/%s.png'%(folder,name), cv2.IMREAD_COLOR)

        if self.mode in ['train']:
            multi_mask = np.load( DATA_DIR + '/image/%s/multi_masks/%s.npy'%(folder,name)).astype(np.int32)
            meta = '<not_used>'

            if self.transform is not None:
                return self.transform(image, multi_mask, meta, index)
            else:
                return input, multi_mask, meta, index

        if self.mode in ['test']:
            if self.transform is not None:
                return self.transform(image, index)
            else:
                return image, index

    def __len__(self):
        return len(self.ids)



# draw  ----------------------------------------------------------------
# def multi_mask_to_overlay_0(multi_mask):
#     overlay = skimage.color.label2rgb(multi_mask, bg_label=0, bg_color=(0, 0, 0))*255
#     overlay = overlay.astype(np.uint8)
#     return overlay

def multi_mask_to_color_overlay(multi_mask, image=None, color=None):

    height,width = multi_mask.shape[:2]
    overlay = np.zeros((height,width,3),np.uint8) if image is None else image.copy()
    num_masks = int(multi_mask.max())
    if num_masks==0: return overlay

    if type(color) in [str] or color is None:
        #https://matplotlib.org/xkcd/examples/color/colormaps_reference.html

        if color is None: color='summer'  #'cool' #'brg'
        color = plt.get_cmap(color)(np.arange(0,1,1/num_masks))
        color = np.array(color[:,:3])*255
        color = np.fliplr(color)
        #np.random.shuffle(color)

    elif type(color) in [list,tuple]:
        color = [ color for i in range(num_masks) ]

    for i in range(num_masks):
        mask = multi_mask==i+1
        overlay[mask]=color[i]
        #overlay = instance[:,:,np.newaxis]*np.array( color[i] ) +  (1-instance[:,:,np.newaxis])*overlay

    return overlay



def multi_mask_to_contour_overlay(multi_mask, image=None, color=[255,255,255]):

    height,width = multi_mask.shape[:2]
    overlay = np.zeros((height,width,3),np.uint8) if image is None else image.copy()
    num_masks = int(multi_mask.max())
    if num_masks==0: return overlay

    for i in range(num_masks):
        mask = multi_mask==i+1
        contour = mask_to_inner_contour(mask)
        overlay[contour]=color

    return overlay

# modifier  ----------------------------------------------------------------

def mask_to_outer_contour(mask):
    pad = np.lib.pad(mask, ((1, 1), (1, 1)), 'reflect')
    contour = (~mask) & (
            (pad[1:-1,1:-1] != pad[:-2,1:-1]) \
          | (pad[1:-1,1:-1] != pad[2:,1:-1])  \
          | (pad[1:-1,1:-1] != pad[1:-1,:-2]) \
          | (pad[1:-1,1:-1] != pad[1:-1,2:])
    )
    return contour

def mask_to_inner_contour(mask):
    pad = np.lib.pad(mask, ((1, 1), (1, 1)), 'reflect')
    contour = mask & (
            (pad[1:-1,1:-1] != pad[:-2,1:-1]) \
          | (pad[1:-1,1:-1] != pad[2:,1:-1])  \
          | (pad[1:-1,1:-1] != pad[1:-1,:-2]) \
          | (pad[1:-1,1:-1] != pad[1:-1,2:])
    )
    return contour

def multi_mask_to_annotation(multi_mask):
    H,W      = multi_mask.shape[:2]
    box      = []
    label    = []
    instance = []

    num_masks = multi_mask.max()
    for i in range(num_masks):
        mask = (multi_mask==(i+1))
        if mask.sum()>1:

            y,x = np.where(mask)
            y0 = y.min()
            y1 = y.max()
            x0 = x.min()
            x1 = x.max()
            w = (x1-x0)+1
            h = (y1-y0)+1


            border = max(2, round(0.2*(w+h)/2))
            #border = max(1, round(0.1*min(w,h)))
            #border = 0
            x0 = x0-border
            x1 = x1+border
            y0 = y0-border
            y1 = y1+border

            #clip
            x0 = max(0,x0)
            y0 = max(0,y0)
            x1 = min(W-1,x1)
            y1 = min(H-1,y1)


            #label
            l = 1 #<todo> support multiclass later ... ?
            if is_small_box_at_boundary((x0,y0,x1,y1),W,H,MIN_SIZE):
                l = IGNORE_BOUNDARY
                continue  #completely ignore!
            elif is_small_box((x0,y0,x1,y1),MIN_SIZE):
                l = IGNORE_SMALL
                continue
            elif is_big_box((x0,y0,x1,y1),MAX_SIZE):
                l = IGNORE_BIG
                continue

            # add --------------------
            box.append([x0,y0,x1,y1])
            label.append(l)
            instance.append(mask)

    box      = np.array(box,np.float32)
    label    = np.array(label,np.float32)
    instance = np.array(instance,np.float32)

    if len(box)==0:
        box      = np.zeros((0,4),np.float32)
        label    = np.zeros((0,1),np.float32)
        instance = np.zeros((0,H,W),np.float32)

    return box, label, instance


def instance_to_multi_mask(instance):

    H,W = instance.shape[1:3]
    multi_mask = np.zeros((H,W),np.int32)

    num_masks = len(instance)
    for i in range(num_masks):
         multi_mask[instance[i]>0] = i+1

    return multi_mask





##------------------------------------------------------
# def instance_to_ellipse(instance):
#  contour = thresh & thresh_to_contour(thresh, radius=1)
#     y,x = np.where(contour)
#     points =  np.vstack((x, y)).T.reshape(-1,1,2)
#
#     #<todo> : see if want to include object that is less than 5 pixel area? near the image boundary?
#     if len(points) >=5:
#         (cx, cy), (minor_d, major_d), angle = cv2.fitEllipse(points)
#         minor_r = minor_d/2
#         major_r = major_d/2
#
#         minor_r = max(1,minor_r)
#         major_r = max(1,major_r)
#
#         #check if valid
#         tx, ty = get_center(thresh)
#         th, tw = thresh.shape[:2]
#         distance = ((cx-tx)*(cx-tx) + (cy-ty)*(cy-ty))**0.5
#         if distance < (major_r+ minor_r)*0.5 and cx>1 and cx<tw-1 and cy>1 and cy <th-1:
#             return cx, cy, minor_r,major_r,  angle
# # r=[8,16,32]
# def multi_mask_to_center(multi_mask, scales=[2,4,8,16]):
#
#     # 3 = math.log2( 8)
#     # 5 = math.log2(32)
#     limits = [  math.log2(s) for s in scales ]
#     limits = [ [c-0.7,c+0.7] for c in limits ]
#     limits[-1][0] = 0
#     limits[ 1][1] = math.inf
#     num = len(limits)
#
#
#     H,W = multi_mask
#     xx, yy = np.meshgrid(np.arange(0, W), np.arange(0, H))
#
#     center = np.zeros((num, H,W), np.bool)
#     delta  = np.zeros((5,H,W), np.float32)  #cx, cy, major_r, minor_r, angle
#
#
#     num_masks = multi_mask.max()
#     for i in range(num_masks):
#         instance = num_masks==(i+1)
#
#         ret = thresh_to_ellipse(thresh)
#         if ret is None: continue
#
#         cx, cy, minor_r, major_r,  angle = ret
#         r = (minor_r + major_r)/2
#         t = math.log2(r)
#
#         cr = max(1,int(0.5*r))
#         c = np.zeros((H,W), np.uint8)
#         cv2.circle(c, (int(cx),int(cy)), int(cr), 255, -1) # cv2.LINE_AA)
#         c = c>128
#
#
#         for j,limit in enumerate(limits):
#             if  t >limit[0] and t<limit[1]:
#                 center[j] = center[j] | c
#
#
#         #delta
#         index = np.where(c)
#         delta[0][index] = xx[index]-cx
#         delta[1][index] = yy[index]-cy
#         delta[2][index] = minor_r
#         delta[3][index] = major_r
#         delta[4][index] = angle
#
#
#     return center, delta



# check ##################################################################################3
def run_check_dataset_reader():

    def augment(image, multi_mask, meta, index):
        box, label, instance = multi_mask_to_annotation(multi_mask)
        return image, multi_mask, box, label, instance, meta, index

    dataset = ScienceDataset(
        'train1_ids_gray2_500', mode='train',
        #'disk0_ids_dummy_9', mode='train',
        #'merge1_1', mode='train',
        transform = augment,
    )
    #sampler = SequentialSampler(dataset)
    sampler = RandomSampler(dataset)


    for n in iter(sampler):
    #for n in range(10):
    #n=0
    #while 1:
        image, multi_mask, box, label, instance, meta, index = dataset[n]

        print('n=%d------------------------------------------'%n)
        print('meta : ', meta)

        contour_overlay = multi_mask_to_contour_overlay(multi_mask,image,color=[0,0,255])
        color_overlay   =   multi_mask_to_color_overlay(multi_mask)
        image_show('image',np.hstack([image,color_overlay,contour_overlay]))

        num_masks  = len(instance)
        for i in range(num_masks):
            x0,y0,x1,y1 = box[i]
            print('label[i], box[i] : ', label[i], box[i])

            instance1 = cv2.cvtColor((instance[i]*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
            image1           = image.copy()
            color_overlay1   = color_overlay.copy()
            contour_overlay1 = contour_overlay.copy()

            cv2.rectangle(instance1,(x0,y0),(x1,y1),(0,255,255),2)
            cv2.rectangle(image1,(x0,y0),(x1,y1),(0,255,255),2)
            cv2.rectangle(color_overlay1,(x0,y0),(x1,y1),(0,255,255),2)
            cv2.rectangle(contour_overlay1,(x0,y0),(x1,y1),(0,255,255),2)
            image_show('instance[i]',np.hstack([instance1, image1,color_overlay1,contour_overlay1]))
            cv2.waitKey(0)





# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_dataset_reader()

    print( 'sucess!')
