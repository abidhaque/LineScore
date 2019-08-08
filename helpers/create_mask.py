import cv2 as cv
import numpy as np

# filename = 'P1'

def create_mask(image):
    kernel = np.ones((5,5),np.uint8)
    print("opencv version = ", str(cv.__version__))
    # im_orig = cv.imread(filename + '.jpg',1)
    # im_gray = cv.cvtColor(im_orig, cv.COLOR_BGR2GRAY)
    im_gray = image

    # blur = cv.MedianBlur(im_gray, 5)
    blur = cv.GaussianBlur(im_gray, (5,5),0)
    dilate = cv.dilate(blur,kernel,iterations = 2)
    erode = cv.erode(dilate,kernel,iterations = 2)



    # ret,mask = cv.threshold(erode,30,255,cv.THRESH_BINARY)
    ret,thresh = cv.threshold(erode,127,255,cv.THRESH_BINARY)
    # mask2 = mask

    # mask = cv.erode(mask, kernel, iterations = 3)
    # mask = cv.adaptiveThreshold(erode,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,81,2)
    # output = cv.bitwise_and(im_gray, im_gray, mask=mask)
    # thresh1 = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,11,2)
    im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # print(contours.shape)
    # cv.imshow("Mask", thresh)
    # cv.waitKey(0)


    if len(contours) != 0:
        # draw in blue the contours that were founded
        cnts = sorted(contours, key = cv.contourArea, reverse = True)
            #find the biggest area
            # print(len(contours))
            # c = max(contours, key = cv.contourArea)
            # im_output = im_gray[c]

            # print(cnts[0])
        # cv.drawContours(output, cnts, 0, 255, 3)

            # x,y,w,h = cv.boundingRect(c)
            # draw the book contour (in green)
            # cv.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)
    # cv.drawContours(im_gray, contours, -1, (0,255,255), 3)
    # cv.imshow("Mask", im_gray)
    # cv.waitKey(0)
    print('Mask Created')
    return cnts, thresh





#
# cv.imshow("Mask", output)
# cv.waitKey(0)
# cv.destroyAllWindows()
