import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import timeit


def print_with_rescale(img_input, scale_percent = 10, window_name = "Titre"):

    print('Original Dimensions : ',img_input.shape)

    width = int(img_input.shape[1] * scale_percent / 100)
    height = int(img_input.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img_input, dim, interpolation = cv2.INTER_AREA)

    print('Resized Dimensions : ',resized.shape)

    cv2.imshow(window_name, resized)
    cv2.waitKey()

def save_image(image, image_name):
    cv2.imwrite("./image_result/RGB_"+image_name, image)

def divide_image(img_input, padding = 100):
    height = int(img_input.shape[0] / 3)
    width = img_input.shape[1]

    img_B = img_input[0:height+2*padding,0:width]
    img_G = img_input[height-padding:2*height+padding,0:width]
    img_R = img_input[2*height-2*padding:3*height,0:width]

    return (img_B,img_G,img_R)

# ---------------------------------------------------------------
# |                                                             |
# |                          Main                               |
# |                                                             |
# ---------------------------------------------------------------
def main():
    # img_name = "00451u.tif"
    # img_name = "00998u.tif"
    img_name = "01520u.tif"
    # img_name = "01887u.tif"

    img = cv2.imread("./"+img_name,0)

    # Divide image
    (image_B, image_G, image_R) = divide_image(img)

    # Merge each channel images
    fusion_temp = (image_B, image_G, image_R)
    img_color = cv2.merge(fusion_temp)
    print_with_rescale(img_color)

    # Save image
    save_image(img_color,img_name)

# ---------------------------------------------------------------

start = timeit.default_timer()
main()
stop = timeit.default_timer()
print('Time :', stop - start, "s")