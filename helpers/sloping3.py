import cv2 as cv
import numpy as np

def find_slope(region,ker):
    # reg = np.array(region, dtype=np.float64)
    kern = np.array(ker, dtype=np.float64)
    divs = kern.shape[0]
    reg = np.ndarray([divs,region.shape[0],region.shape[1]], dtype=np.float64)
    reg[:] = np.array(region, dtype=np.float64)

    slope_max = np.amax((np.sum(np.multiply(reg,kern==255), axis=(1,2))/np.count_nonzero(kern==255, axis=(1,2)) - np.sum(np.multiply(reg,kern==0), axis=(1,2))/np.count_nonzero(kern==0, axis=(1,2)))/255)
    slope_min = np.amin((np.sum(np.multiply(reg,kern==255), axis=(1,2))/np.count_nonzero(kern==255, axis=(1,2)) - np.sum(np.multiply(reg,kern==0), axis=(1,2))/np.count_nonzero(kern==0, axis=(1,2)))/255)

    #slope_max + (slope_max-slope_min)
    # diff = 2*slope_max-slope_min
    diff = slope_max-slope_min
    max_slope_angle = np.argmax((np.sum(np.multiply(reg,kern==255), axis=(1,2))/np.count_nonzero(kern==255, axis=(1,2)) - np.sum(np.multiply(reg,kern==0), axis=(1,2))/np.count_nonzero(kern==0, axis=(1,2)))/255)

    max_diff_angle = np.argmax(((np.sum(np.multiply(reg,kern==255), axis=(1,2))/np.count_nonzero(kern==255, axis=(1,2)) - np.sum(np.multiply(reg,kern==0), axis=(1,2))/np.count_nonzero(kern==0, axis=(1,2)))/255) - ((np.sum(np.multiply(reg,kern==255), axis=(1,2))/np.count_nonzero(kern==255, axis=(1,2)) - np.sum(np.multiply(reg,kern==0), axis=(1,2))/np.count_nonzero(kern==0, axis=(1,2)))/255))


    return(slope_max, max_slope_angle,diff,max_diff_angle)
