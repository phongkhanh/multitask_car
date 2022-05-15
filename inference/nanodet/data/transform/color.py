import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

def random_brightness(img, delta):
    img += random.uniform(-delta, delta)
    return img


def random_contrast(img, alpha_low, alpha_up):
    img *= random.uniform(alpha_low, alpha_up)
    return img


def random_saturation(img, alpha_low, alpha_up):
    hsv_img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)

    hsv_img[..., 1] *= random.uniform(alpha_low, alpha_up)

    img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return img

def random_hue(img, alpha_low, alpha_up):
    hsv_img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)

    hsv_img[..., 0] *= random.uniform(alpha_low, alpha_up)

    img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return img
def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

def normalize(meta, mean, std):
    img = meta['img'].astype(np.float32)
    mean = np.array(mean, dtype=np.float64).reshape(1, -1)
    stdinv = 1 / np.array(std, dtype=np.float64).reshape(1, -1)
    cv2.subtract(img, mean, img)
    cv2.multiply(img, stdinv, img)
    meta['img'] = img
    return meta


def _normalize(img, mean, std):
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3) / 255
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3) / 255
    img = (img - mean) / std
    return img


def color_aug_and_norm(meta, kwargs):
    #print(meta['img'].dtype)
    count=random.randint(1,7000)
    img = meta['img'].astype(np.float32) / 255
    if 'brightness' in kwargs and random.randint(0, 1):
        img = random_brightness(img, kwargs['brightness'])

    if 'contrast' in kwargs and random.randint(0, 1):
        img = random_contrast(img, *kwargs['contrast'])

    if 'saturation' in kwargs and random.randint(0, 1):
        img = random_saturation(img, *kwargs['saturation'])
    # if 'hue' in kwargs and random.randint(0, 1):
    #     img = random_hue(img, *kwargs['hue'])
 
    #cv2.imshow('trans', (img*255).astype('uint8'))
    #cv2.imwrite('/content/drive/MyDrive/save_img/frame%d.jpg' %count,img*255)
    #cv2.waitKey(0)
    #plt.imshow((img*255).astype('unint8'))
   # plt.show()
    img = _normalize(img, *kwargs['normalize'])
    #plt.imshow((img*255).astype('float32'))
    #plt.show()
    meta['img'] = img
    # print(meta['img'].dtype)
    return meta


