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
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs])) # return mutual information scalar

def gradient(img):
    # filtre de Sobel
    kernel = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], float)
    dx = kernel
    dy = np.transpose(kernel)

    # return Gx , Gy
    return cv2.filter2D(img,-1,dx), cv2.filter2D(img,-1,dy)

def recalage(img_model, img_recal):
    img_translation = img_recal.copy()
    pi = 0.0
    pi_old = 1.0
    qi_old = 1.0
    qi = 0.0

    epsilon = 0.0000001

    # mut_info = mutual_information(img_model, img_translation)

    while(abs(pi_old - pi) > 0.1 and abs(qi_old - qi) > 0.1):
        pi_old = pi
        qi_old = qi
        gx, gy = gradient(img_translation)

        dmut_info_dp = float(2.0 * np.sum(np.sum(np.subtract(img_translation, img_model) * gx)))
        dmut_info_dq = float(2.0 * np.sum(np.sum(np.subtract(img_translation, img_model) * gy)))

        pi = pi - epsilon * dmut_info_dp
        qi = qi - epsilon * dmut_info_dq

        img_translation = translation(img_recal, -pi, -qi)

        cv2.imshow("Recalage", img_translation)
        cv2.waitKey(10)

        # print("pi = " + str(pi) + " ; qi = " + str(qi))
        # print("pi_old - pi = " + str(abs(pi_old - pi)) + " ; qi_old - qi = " + str(abs(qi_old - qi)))

        # mut_info = mutual_information(img_model, img_translation)

    return img_translation

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
    
    # Debug area ---------------------------------------------------------------------
    width = int(image_B.shape[1] * 10 / 100)
    height = int(image_B.shape[0] * 10 / 100)
    dim = (width, height)
    # resize image
    image_B_rescale = cv2.resize(image_B, dim, interpolation = cv2.INTER_AREA)
    image_B_rescale, _ = gradient(image_B_rescale)

    width = int(image_R.shape[1] * 10 / 100)
    height = int(image_R.shape[0] * 10 / 100)
    dim = (width, height)
    # resize image
    image_R_rescale = cv2.resize(image_R, dim, interpolation = cv2.INTER_AREA)
    image_R_rescale, _ = gradient(image_R_rescale)

    width = int(image_G.shape[1] * 10 / 100)
    height = int(image_G.shape[0] * 10 / 100)
    dim = (width, height)
    # resize image
    image_G_rescale = cv2.resize(image_G, dim, interpolation = cv2.INTER_AREA)
    image_G_rescale, _ = gradient(image_G_rescale)

    print(mutual_information(image_B_rescale,image_G_rescale))
    img_recale = recalage(image_B_rescale,image_G_rescale)
    print(mutual_information(image_B_rescale,img_recale))

    fusion = (image_B_rescale, img_recale, image_R_rescale)
    cv2.imshow("test 1..22...2.22.", cv2.merge(fusion))
    cv2.waitKey()
    
    # ---------------------------------------------------------------------------------

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
