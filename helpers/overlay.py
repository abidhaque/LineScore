import cv2 as cv
import numpy as np
# import convolve
# import convolve


def make_pad(image,pad):
    image = cv.copyMakeBorder(image, pad, pad, pad, pad,
        cv.BORDER_REPLICATE)
    return image


alpha = 0.5
kernel_size = 9
pad = (kernel_size-1)//2
point = 'P4'
orgfilename = point + '.jpg'
colormaskname = 'P4_output_kersize9_elemsize9_widthrange2' + '.jpg'
print(colormaskname)

img1 = cv.imread(orgfilename)
img1 = make_pad(img1,pad)
img2 = cv.imread(colormaskname)
overlay = img1.copy()
output = img2.copy()
print(img1.shape)

#
print(img2.shape)
#
# # output = np.zeros_like(img2, dtype=np.uint8)
#
cv.addWeighted(overlay, alpha, output, 1 - alpha,0, output)
outfilename = point + '_overlay' + '.jpg'
cv.imwrite(outfilename, output)

cv.imshow("Mask", output)
cv.waitKey(0)
cv.destroyAllWindows()
