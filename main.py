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

def translation(img,x,y):
    heigth,width = img.shape

    M = np.float32([[1,0,x],[0,1,y]])
    dst = cv2.warpAffine(img,M,(width,heigth))

    return dst

def rotation(img,angle):
    heigth,width = img.shape

    M = cv2.getRotationMatrix2D((width/2,heigth/2),angle,1)
    dst = cv2.warpAffine(img,M,(width,heigth))

    return dst

def histogram_joint(img1,img2,print_histo=False):
    hist_2d, _, _ = np.histogram2d(img1.ravel(), img2.ravel(), bins=20)
    
    if(print_histo):
        hist_2d_log = np.zeros(hist_2d.shape)
        non_zeros = hist_2d != 0
        hist_2d_log[non_zeros] = np.log(hist_2d[non_zeros])
        plt.imshow(hist_2d_log.T, origin='lower')
        plt.show()

    return hist_2d

def mutual_information(img1,img2):
    hist2D = histogram_joint(img1,img2)
    pxy = hist2D / float(np.sum(hist2D))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def gradient(img):
    # filtre de Sobel
    kernel = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], float)
    dx = kernel
    dy = np.transpose(kernel)

    return cv2.filter2D(img,-1,dx), cv2.filter2D(img,-1,dy)

    
# ---------------------------------------------------------------
# |                                                             |
# |                          Main                               |
# |                                                             |
# ---------------------------------------------------------------
def main():
    # img_name = "00451u.tif" #Ville
    # img_name = "00998u.tif" #Eglise
    # img_name = "01520u.tif" #Potager
    img_name = "01887u.tif" #Russe

    img = cv2.imread("./"+img_name,0)

    # Divide image
    (image_B, image_G, image_R) = divide_image(img)

    mutual_information(image_B,image_G)
    # Merge each channel images
    fusion_temp = (image_B, image_G, image_R)
    
    img_color = cv2.merge(fusion_temp)
    # print_with_rescale(img_color)

    # Save image
    save_image(img_color,img_name)

# ---------------------------------------------------------------

start = timeit.default_timer()
main()
stop = timeit.default_timer()
print('Time :', stop - start, "s")


"""
TODO

faire comme matlab
grad_centre => filtre de sobel


"""
