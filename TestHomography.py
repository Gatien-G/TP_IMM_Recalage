import cv2
import numpy as np

def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def get_gradient(im) :
    # Calculate the x and y gradients using Sobel operator
    grad_x = cv2.Sobel(im,cv2.CV_32F,1,0,ksize=3)
    grad_y = cv2.Sobel(im,cv2.CV_32F,0,1,ksize=3)
 
    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad

# Read 8-bit color image.
# This is an image in which the three channels are
# concatenated vertically.
#-----------------------------------------------
im =  cv2.imread("01887u.tif",0) #Russe
# im =  cv2.imread("00451u.tif",0) #ville
# im =  cv2.imread("00998u.tif",0) #Eglise
# im =  cv2.imread("01520u.tif",0) #Potager

im = cv2.resize(im,None, fx=0.1, fy=0.1)
# im = cv2.resize(im,None, fx=0.6, fy=0.6)

# im =  cv2.imread("emir.jpg",0)
#-------------------------------------------------

mode = "sift"
# mode = "findECC"

# Define motion model
warp_mode = cv2.MOTION_HOMOGRAPHY
# warp_mode = cv2.MOTION_AFFINE
#--------------------------------------------------

# Find the width and height of the color image
sz = im.shape
print(sz)
height = int(sz[0] / 3)
width = sz[1]

# Extract the three channels from the gray scale image
# and merge the three channels into one color image
crop = 0
im_color = np.zeros((height-2*crop,width-2*crop,3), dtype=np.uint8 )

for i in range(0,3) :
    img_color_tmp = im[i * height:(i+1) * height,:]

    sz = img_color_tmp.shape

    im_color[:,:,i] = img_color_tmp[crop:sz[0]-crop, crop:sz[1]-crop]
    cv2.imshow("crop",im_color[:,:, i])
    cv2.waitKey(0)

cv2.destroyAllWindows()

#--------HOMOGRAPHY C PARTI---------------

# Find the width and height of the cropped color image
sz = im_color.shape
print(sz)
height = sz[0]
width = sz[1]

# Allocate space for aligned image
im_aligned = np.zeros((height,width,3), dtype=np.uint8 )

# The blue and green channels will be aligned to the red channel.
# So copy the red channel
im_aligned[:,:,2] = im_color[:,:,2]

# Set the warp matrix to identity.
warp_matrix = np.eye(3, 3, dtype=np.float32)

if(mode == "findECC"):

    # Set the warp matrix to identity.
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Set the stopping criteria for the algorithm.
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000,  1e-10)

    # Warp the blue and green channels to the red channel
    for i in range(0,2) :
        (cc, warp_matrix) = cv2.findTransformECC(get_gradient(im_color[:,:,2]), get_gradient(im_color[:,:,i]),warp_matrix, warp_mode, criteria, None, 5)

        if warp_mode == cv2.MOTION_HOMOGRAPHY :
            # Use Perspective warp when the transformation is a Homography
            im_aligned[:,:,i] = cv2.warpPerspective(im_color[:,:,i], warp_matrix, (width,height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            # Use Affine warp when the transformation is not a Homography
            im_aligned[:,:,i] = cv2.warpAffine(im_color[:,:,i], warp_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        print(warp_matrix)

elif(mode == "sift"):
    orb = cv2.ORB_create()

    # create BFMatcher object based on hamming distance  
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # find the keypoints and descriptors with SIFT
    kp_model, des_model =  orb.detectAndCompute(im_color[:,:,2],None)



    for i in range(0,2):
        kp_frame, des_frame = orb.detectAndCompute(im_color[:,:,i],None)
        # match frame descriptors with model descriptors
        matches = bf.match(des_model, des_frame)
        # sort them in the order of their distance
        # the lower the distance, the better the match
        matches = sorted(matches, key=lambda x: x.distance)


        if len(matches) > 10:
            display_matches = cv2.drawMatches(im_color[:,:,2], kp_model, im_color[:,:,i], kp_frame, matches, 0, flags=2)
            cv2.imshow("matches", display_matches)
            cv2.waitKey()
            # differenciate between source points and destination points
            src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
          
            if warp_mode == cv2.MOTION_HOMOGRAPHY :
                # compute Homography
                warp_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                # K = np.float64([[width, 0, 0.5*(width-1)],
                #                [0, width, 0.5*(height-1)],
                #                [0.0,0.0,1.0]])
                # num, Rs, Ts, Ns  = cv2.decomposeHomographyMat(homography, K)
                # print(Ts)

                im_aligned[:,:,i] = cv2.warpPerspective(im_color[:,:,i], warp_matrix, (width,height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

            else:
                if warp_mode == cv2.MOTION_AFFINE:
                    #Get the Affine Transform
                    warp_matrix = cv2.getAffineTransform(src_pts[7:10], dst_pts[7:10])
                # Use Affine warp when the transformation is not a Homography
                im_aligned[:,:,i] = cv2.warpAffine(im_color[:,:,i], warp_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            
            print(warp_matrix)
                    
        else:
            print("not enough matches")

cv2.destroyAllWindows()
# Show final output
cv2.imshow("img de base",im_color)
cv2.waitKey(0)
cv2.imshow("aligned img",im_aligned)
cv2.waitKey(0)

#test auto white balance 
cv2.imshow("auto wp aligned img", white_balance(im_aligned))
cv2.waitKey(0)
#equalized histogram for white balance
for i in range(0,3):
    im_aligned[:,:,i] = cv2.equalizeHist(im_aligned[:,:,i])

cv2.imshow("equHist aligned img",im_aligned)
cv2.waitKey(0)
cv2.destroyAllWindows()
