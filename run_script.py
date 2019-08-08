import cv2 as cv
import numpy as np
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
from colorspacious import cspace_converter
import sys
import os
from skimage.exposure import rescale_intensity
import multiprocessing as mp
from helpers import create_mask, maxmin, mkernels_255, sloping3, colormasking



startTime = datetime.now()
print(str(startTime))
# filename = 'IP-Create Image Subset-47_c1'
filename = sys.argv[1]



# filename1 = 'P6'
filename1 = filename + '_highcontrast'

# img = cv.imread(filename + '.jpg',1)
img1 = cv.imread(filename1 + '.jpg',0)
img = cv.imread(filename + '.jpg',0)

# outputimgpath = filename + '_output' + '.jpg'

#grayscale conversion
im_gray1 = img1
im_gray = img

# print(im_gray.shape)


#Image dimensions
print(im_gray.shape)


kernel_size = 5                         #Kernel dimensions
window_size = kernel_size               #Optional smaller window inside kernel
divisions = 2*kernel_size-2             #Number of angle divisions
maxmin_value_slope = []                       #Output List
maxmin_value_diff = []
# width = 2
width_range = [1]
element_size = kernel_size
y_0,y_1 = 188,296
x_0,x_1 = 208,315


pad = (kernel_size-1)//2                #Padding image with zeros
(iH, iW) = img.shape[:2]
# dst = cv.fastNlMeansDenoisingColored(img,None,10,10,7,21)

#Padding Function
def make_pad(image,pad):
    image = cv.copyMakeBorder(image, pad, pad, pad, pad,
        cv.BORDER_REPLICATE)
    return image
im_gray1 = make_pad(im_gray1,pad)
im_gray = make_pad(im_gray,pad)
# im_gray = cv.GaussianBlur(im_gray,(3,3),0)
im_contour = np.zeros_like(im_gray)
colormask_slope = np.zeros_like(im_gray, dtype = np.float64)
colormask_diff = np.zeros_like(im_gray, dtype = np.float64)

color_rescaled_slope = np.zeros_like(colormask_slope, dtype = np.uint8)
color_rescaled_diff = np.zeros_like(colormask_diff, dtype = np.uint8)


ker = np.zeros([len(width_range),divisions, kernel_size, kernel_size], dtype=np.uint8)
bright_kernel = 255*np.ones(ker.shape, dtype=np.float64)


for i in range(len(width_range)):
    ker[i] = mkernels_255.make_rotating_kernels(kernel_size,window_size,width_range[i],divisions,element_size)


#Create Mask using blurring and thresholding
cnts,mask = create_mask.create_mask(im_gray1)
cnts,mask = create_mask.create_mask(im_gray1)
maskname = filename + '_mask' + '.jpg'
cv.imwrite(maskname, mask)

output = cv.bitwise_and(im_gray, im_gray, mask=mask)
response = np.zeros(len(width_range))
max_angle = np.zeros(len(width_range))

points_list = []



for y in np.arange(pad, iH + pad):
    for x in np.arange(pad, iW + pad):
        if cv.pointPolygonTest(cnts[0],(x,y),False) == 1:
            points_list.append((y,x))

print(np.array(points_list))

def loop_func(point_tuple, im_gray=im_gray, colormask_slope=colormask_slope, colormask_diff=colormask_diff, maxmin_value_slope=maxmin_value_slope, maxmin_value_diff=maxmin_value_diff):
    (y,x) = point_tuple
    roi = im_gray[y - pad:y + pad + 1, x - pad:x + pad + 1]
    if np.count_nonzero(roi) > 0:
        rescaled_intermediate = np.zeros([im_gray.shape[0], im_gray.shape[1], roi.shape[0], roi.shape[1]], dtype=np.uint8)
        max0 = np.amax(roi)
        min0 = np.amin(roi)
        if max0-min0==0:
            rescaled_intermediate[y,x] = max0
        else:
            rescaled_intermediate[y,x] = roi

        slope_response = np.zeros(len(width_range))
        diff_response = np.zeros(len(width_range))
        slope_angle = np.zeros(len(width_range))
        diff_angle = np.zeros(len(width_range))

        for i in range(len(width_range)):
            signal = 0
            slope_signal,slope_index,diff_signal,diff_index = sloping3.find_slope(rescaled_intermediate[y,x],ker[i])
            # print(signal)
            slope_response[i] = slope_signal
            diff_response[i] = diff_signal
            slope_angle[i] = slope_index
            diff_angle[i] = diff_index

        responses = np.array([slope_response,slope_angle,diff_response,diff_angle])
        colormask_slope[y,x] = colormasking.append_colormask(responses[0:2],ker)
        colormask_diff[y,x] = colormasking.append_colormask(responses[2:4],ker)
    else:
        colormask_slope[y,x] = 0
        colormask_diff[y,x] = 0
    # print(colormask_slope[y,x], colormask_diff[y,x])
    maxmin_value_slope.append((colormask_slope[y,x]))
    maxmin_value_diff.append((colormask_diff[y,x]))

    return(point_tuple, colormask_slope[y,x], colormask_diff[y,x])
#
pool = mp.Pool(mp.cpu_count())
result = pool.map(loop_func, [point for point in points_list])

print("Convolution Completed!")

for point in result:
    # print(point)
    point_tuple = point[0]
    color_slope = point[1]
    color_diff = point[2]
    colormask_slope[point_tuple[0], point_tuple[1]] = color_slope
    colormask_diff[point_tuple[0], point_tuple[1]] = color_diff
    maxmin_value_slope.append((color_slope))
    maxmin_value_diff.append((color_diff))


newdir = "./" + filename
os.mkdir(newdir)
os.chdir(newdir)
maxmin_value_slope = np.array(maxmin_value_slope)
maxmin_value_diff = np.array(maxmin_value_diff)



filepath_slope = filename + '_slope' + '_kersize' + str(kernel_size) + '_elemsize' + str(element_size) + '_widthrange' + str(width_range)  + '.csv'
filepath_diff = filename + '_diff' + '_kersize' + str(kernel_size) + '_elemsize' + str(element_size) + '_widthrange' + str(width_range)  + '.csv'

np.savetxt(filepath_slope, colormask_slope, delimiter=",")
np.savetxt(filepath_diff, colormask_diff, delimiter=",")

min_slope = np.amin(colormask_slope)
# print(min)
max_slope = np.amax(colormask_slope)
min_diff = np.amin(colormask_diff)
max_diff = np.amax(colormask_diff)


print(min_slope, max_slope, min_diff, max_diff)

# min = np.minimum(min_slope,min_diff)
# max = np.maximum(max_slope,max_diff)

print(min_slope,max_slope,min_diff,max_diff)

color_rescaled_slope = np.zeros_like(colormask_slope, dtype=np.uint8)
color_rescaled_diff = np.zeros_like(colormask_diff, dtype=np.uint8)

color_rescaled_slope[:,:] = ((colormask_slope[:,:]-min_slope)/(max_slope-min_slope))*255
color_rescaled_diff[:,:] = ((colormask_diff[:,:]-min_diff)/(max_diff-min_diff))*255

# color_rescaled[color_rescaled<125] = 0

#output file path
myfile_path_slope = filename + '_slope_maxmin' + '_kersize' + str(kernel_size) + '_elemsize' + str(element_size) + '_widthrange' + str(width_range)  + '.csv'
myfile_path_diff = filename + '_diff_maxmin' + '_kersize' + str(kernel_size) + '_elemsize' + str(element_size) + '_widthrange' + str(width_range)  + '.csv'
filepath_8bit_slope = filename + '_slope_8bit'+ '_kersize' + str(kernel_size) + '_elemsize' + str(element_size) + '_widthrange' + str(width_range) + '.csv'
filepath_8bit_flat_slope = filename + '_slope_8bit_flat' + '_kersize' + str(kernel_size) + '_elemsize' + str(element_size) + '_widthrange' + str(width_range)  + '.csv'
filepath_8bit_diff = filename + '_diff_8bit' + '_kersize' + str(kernel_size) + '_elemsize' + str(element_size) + '_widthrange' + str(width_range)  + '.csv'
filepath_8bit_flat_diff = filename + '_diff_8bit_flat' + '_kersize' + str(kernel_size) + '_elemsize' + str(element_size) + '_widthrange' + str(width_range)  + '.csv'

np.savetxt(filepath_8bit_slope, color_rescaled_slope, delimiter=",")
np.savetxt(filepath_8bit_diff, color_rescaled_diff, delimiter=",")
np.savetxt(filepath_8bit_flat_slope, color_rescaled_slope.flatten(), delimiter=",")
np.savetxt(filepath_8bit_flat_diff, color_rescaled_diff.flatten(), delimiter=",")

rescaled_gray_slope = cv.cvtColor(color_rescaled_slope,cv.COLOR_GRAY2BGR)
rescaled_gray_diff = cv.cvtColor(color_rescaled_diff,cv.COLOR_GRAY2BGR)

# color_rescaled2 = np.zeros([color_rescaled.shape[0],color_rescaled.shape[1],3], dtype=np.uint8)
# color_rescaled2[:,:,2] = color_rescaled[:]
lookup = np.zeros([256,1,3], dtype=np.uint8)
print(lookup.shape)

for i in range(lookup.shape[0]):
    lookup[i,0] = [i,i,i]

# color_rescaled2 = cv.LUT(rescaled_gray,lookup)

color_rescaled2_slope = cv.applyColorMap(color_rescaled_slope, cv.COLORMAP_JET)
color_rescaled2_diff = cv.applyColorMap(color_rescaled_diff, cv.COLORMAP_JET)

np.savetxt(myfile_path_slope, maxmin_value_slope, delimiter=",")
np.savetxt(myfile_path_diff, maxmin_value_diff, delimiter=",")

print("time taken = " + str(datetime.now() - startTime))
# outputimgpath = filename + '_output' + '_kersize' + str(kernel_size) + '_elemsize' + str(element_size) + '_width' + str(width) + '.jpg'
outputimgpath_slope = filename + '_slope' + '_kersize' + str(kernel_size) + '_elemsize' + str(element_size) + '_widthrange' + str(width_range) + '.jpg'
outputimgpath_diff = filename + '_diff' + '_kersize' + str(kernel_size) + '_elemsize' + str(element_size) + '_widthrange' + str(width_range) + '.jpg'

sys.stdout.write('\a')
sys.stdout.flush()
cv.imwrite(outputimgpath_slope, color_rescaled2_slope)
cv.imwrite(outputimgpath_diff, color_rescaled2_diff)
