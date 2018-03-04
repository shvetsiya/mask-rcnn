from common import *


def prob_delta_to_candidates( prob, delta, heads, threshold=0.4):
    num_heads =len(heads)

    candidates = []
    for h in range(num_heads):
        index = np.where(prob[h]>threshold)
        if len(index[0])!=0:
            y,x      = index[0],index[1]
            dx       = delta[h,0][index]
            dy       = delta[h,1][index]
            dminor_r = delta[h,2][index]
            dmajor_r = delta[h,3][index]
            sin      = delta[h,4][index]
            cos      = delta[h,5][index]

            r  = heads[h]
            rr = max(1,int(0.33*2**r))

            cx      = (dx*rr+x)
            cy      = (dy*rr+y)
            minor_r = (2**(dminor_r+r))
            major_r = (2**(dmajor_r+r))
            angle   = (np.arctan2(sin,cos)/np.pi*180)
            score   = prob[h][index]

            c = np.vstack((cx,cy,minor_r,major_r,angle,score))
            candidates.append(c.T)

    candidates = np.concatenate(candidates)
    return candidates

# https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
def non_max_suppress(candidates, min_distance_threshold=0.25):

    if len(candidates) == 0:
        return []

    cx     =  candidates[:,0]
    cy     =  candidates[:,1]
    log2r  = np.log2 ( (candidates[:,2] + candidates[:,3])/2)
    score  =  candidates[:,5]
    indices = np.argsort(-score)  #decreasing

    select = []
    while len(indices) > 0:
        i = indices[0]
        select.append(i)

        # last added
        distances = ((cx[indices] - cx[i])**2 \
                +    (cy[indices] - cy[i])**2 \
                +    (log2r[indices] - log2r[i])**2)**0.5


        # delete all  candidates that is nearby
        #remove = np.where(distances < min_distance_threshold*log2r[i])
        remove = np.where(distances < 6)  #<todo> a good thresholding mnethods
        indices = np.delete( indices, remove )

    nms = candidates[select]
    return nms

def nms_to_original_size( nms, image, original_image):
    h,w = image.shape[:2]
    original_h,original_w = original_image.shape[:2]

    scale_x = original_w/w
    scale_y = original_h/h
    nms[:,0] *= scale_x
    nms[:,1] *= scale_y
    nms[:,2] *= scale_x
    nms[:,3] *= scale_y

    return nms


def nms_to_label( nms, image ):

    H,W = image.shape[:2]
    label = np.zeros((H,W,3), np.uint8)

    num_mns = len(nms)
    assert(num_mns<256*256)

    nms = nms.astype(np.int32)
    for i in range(num_mns):
        candidate = nms[i]
        b,g,r = 0, (i+1)//256, (i+1)%256
        cx,cy, minor_r,major_r, angle, score = candidate
        cv2.ellipse(label, (cx,cy), (minor_r,major_r), angle, 0, 360, (b,g,r), -1)

    label = label.astype(np.int32)
    label = label[:,:,1]*256 + label[:,:,2]

    return label