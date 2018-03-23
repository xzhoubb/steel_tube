import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
%matplotlib inline


def load_img(img_dir):
    img_names = os.listdir(img_dir)
    img_paths = [img_dir + img_name for img_name in img_names if img_name.endswith(".bmp")]
    imgs = [cv2.imread(path, 0) for path in img_paths]
    shapes = [img.shape for img in imgs]
    return imgs, img_paths, shapes


def gradient_transform(img_gray): #paramter tuning
    img_gray = cv2.GaussianBlur(img_gray,(21,21),0)
    img_gray = cv2.equalizeHist(img_gray) 
    rect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    gradient = cv2.morphologyEx(img_gray, cv2.MORPH_GRADIENT, rect)  
    upper = 100
    lower = 50
    uw = gradient > upper
    gradient[uw] = 255
    lw = gradient < lower
    gradient[lw] = 0    
    return gradient


def show(img):
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    return None


def auto_grad(img):
    img = cv2.GaussianBlur(img,(21,21),0)
    img = cv2.equalizeHist(img)
    # compute the median of the single channel pixel intensities
    rect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, rect)
    v = np.median(img)
    sigma = 0.33

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    
    uw = gradient > upper
    gradient[uw] = 255
    lw = gradient < lower
    gradient[lw] = 0
    return gradient


def opening(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(12,12))
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return opening


def closing(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(12,12))
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return closing



# batch process
img_dir = "/Users/zhouxiaoyang/Desktop/Lenovo-图像/080323/hg_seg/"
for sub_dir in next(os.walk(img_dir))[1]:
    img_sub_dir = os.path.join(img_dir, sub_dir)
    img_sub_dir += "/"
    imgs, img_paths, shapes = load_img(img_sub_dir)
    crops = [auto_grad(img) for img in imgs]
    crops = [closing(img) for img in crops]
    crops = [opening(img) for img in crops]
    for i, crop in enumerate(crops):
        cv2.imwrite(img_sub_dir+img_paths[i].split("/")[-1].split(".")[0]+"_grad"+".bmp", crop)