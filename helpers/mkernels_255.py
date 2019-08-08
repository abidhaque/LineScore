import numpy as np
import math
import cv2 as cv

def make_rotating_kernels(kernel_size,window_size,thicc,divisions,element_size):
    ker_size = kernel_size
    win_size = kernel_size
    divisions = 2*ker_size - 2
    width = thicc
    el_size = element_size
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE,(el_size,el_size))
    element = np.pad(element,(ker_size-el_size)//2,'constant', constant_values=0)
    # print(element)


    #Inner box
    win = np.zeros([divisions, win_size, win_size], dtype=np.uint8)
    # print(win.shape)
    #Outer Box
    ker = np.zeros([divisions, ker_size, ker_size], dtype=np.uint8)
    # print(ker.shape)
    # ellipse_kernel =  cv.getStructuringElement(cv.MORPH_ELLIPSE,(ker_size,ker_size))
    center = (win_size-1)/2

    x1 = 0
    y1 = 0
    x2 = ker_size-1
    y2 = ker_size-1
    # temp = win[0,0]


    for i in range(ker_size-1):
        win[i] = np.zeros([ker_size,ker_size], dtype=np.uint8)
        win[i] = cv.line(win[i], (x1,y1+i), (x2,y2-i),(1,0,0),width)
        # print(win[i])

    for i in range(ker_size-1):
        win[ker_size-1+i] = np.zeros([ker_size,ker_size], dtype=np.uint8)
        win[ker_size-1+i] = cv.line(win[ker_size-1+i], (x1+i,ker_size-1), (x2-i,0),(1,0,0),width)
        # print(win[ker_size-1+i])

    for i in range(divisions):
        ker[i] = cv.bitwise_and(win[i], element, mask=element)
        ker[i] = 255*ker[i]
        # ker[i] = cv.GaussianBlur(ker[i],(5,5),0)
        # ker[i] = np.pad(win[i],(ker_size-win_size)/2,'constant', constant_values=0)
        # ker[i] = win[i]
        print(ker[i])

    # print(win.shape)
    # print(ker.shape)
    # print(ker)
    print('Kernels Created')
    return(ker)



# print(win[:,0])
