from common import *

def run_length_encode(x):
    bs = np.where(x.T.flatten() == 1)[0]

    rle = []
    prev = -2
    for b in bs:
        if (b>prev+1): rle.extend((b + 1, 0))
        rle[-1] += 1
        prev = b

    rle = ' '.join([str(r) for r in rle])
    return rle


#https://www.kaggle.com/wcukierski/example-metric-implementation
def compute_average_precision(predict, label, t_range=np.arange(0.5, 1.0, 0.05)):

    # Precision helper function
    def compute_precision(threshold, iou):
        matches = iou > threshold
        true_positives  = np.sum(matches, axis=1) == 1  # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn



    num_label   = len(np.unique(label  ))
    num_predict = len(np.unique(predict))


    # Compute intersection between all objects
    intersection = np.histogram2d(label.flatten(), predict.flatten(), bins=(num_label, num_predict))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(label,   bins = num_label  )[0]
    area_pred = np.histogram(predict, bins = num_predict)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred,  0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    precision = []
    average_precision = 0
    for t in t_range:
        tp, fp, fn = compute_precision(t, iou)
        p = tp / (tp + fp + fn)
        precision.append((t, p, tp, fp, fn))
        average_precision += p

    average_precision /= len(precision)
    return average_precision, precision

def print_precision(precision):

    print('thresh   prec    TP    FP    FN')
    print('---------------------------------')
    for (t, p, tp, fp, fn) in precision:
        print('%0.2f     %0.2f   %3d   %3d   %3d'%(t, p, tp, fp, fn))

#################################################3
## post process ###
def filter_small(label, threshold):
    num = label.max()

    j=0
    for i in range(num):
        thresh = (label==(i+1))
        label[thresh]=(j+1)

        area = thresh.sum()
        if area < threshold:
            label[thresh]=0
        else:
            j = j+1

    return label
