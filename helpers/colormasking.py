import numpy as np
import cv2 as cv
from helpers import sloping3

def append_colormask(response_array,ker):
    resp = np.array(response_array)
    signal_value, signal_index = resp[0], resp[1]

    # difference = maxmin.maxmin(roi,ker)
    # print(np.argmax(signal_value),int(signal_index[np.argmax(signal_value)]))
    response = np.amax(signal_value)
    # print(np.argmax(signal_value),signal_index[np.argmax(signal_value)])

    winner = ker[np.argmax(signal_value),int(signal_index[np.argmax(signal_value)])]
    # max_diff, average_intensity = maxmin.maxmin(winner,ker[np.argmax(response)])
    max_signal, a,b,c = sloping3.find_slope(winner,ker[np.argmax(signal_value)])
    # max_diff = np.sum(np.multiply(winner,roi))
    # print(max_diff)
    # colormask[y,x] = difference
    if max_signal != 0:
        colormask_val = (response/max_signal)

    return(colormask_val)
