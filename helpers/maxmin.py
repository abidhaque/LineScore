# import gen_kernels
import cv2 as cv
import numpy as np
from skimage.draw import line

def maxmin(roi,kernel):
    divs = kernel.shape[0]
    kernel = np.array(kernel,dtype=np.int8)
    region = np.array(roi,dtype=np.uint8)
    # print(kernel)
    response = np.zeros([divs])
    # norm_response = np.zeros([divs])
    ker_size = len(kernel[1])
    # win_size = len(window[1])
    # filter_pad = (ker_size-win_size)//2
    max = 0
    min = 0

    # Iavg,std = cv.meanStdDev(roi)
    for i in range(divs):

        multiply = np.multiply(region,kernel[i])
        positives_avg = 0
        negatives_avg = 0
        # norm_multiply = np.multiply(kernel[i],kernel[i])
        # print(multiply)
        if np.count_nonzero(multiply)>0:
            if len(multiply[multiply>0]>0):
                positives_avg = np.average(multiply[multiply>0])
            if len(multiply[multiply<0]>0):
                negatives_avg = np.average(multiply[multiply<0])
            response[i] = positives_avg+negatives_avg
            # norm_response[i] = np.sum(norm_multiply)
            # response[i] = np.sum(multiply)
        else:
            response[i] = 0

    max = np.amax(response)
    # norm_max = np.amax(norm_response)

    # print(max)
    max_angle = np.argmax(response)
    min = np.amin(response)
    # norm_min = np.amin(norm_response)
    diff = (max - min)
    # norm_diff = norm_max - norm_min
    # print(diff)

    return diff,max_angle
