import cv2
import numpy as np
import skimage.measure

def rgb2gray(img):
    '''
    input img  (H, W, 3)
    '''
    grey = ((img[:,:,0]*0.2989 + img[:,:,1]*0.687 + img[:,:,2]*0.114))
    #cv2.imwrite("input_grey.jpg",(grey*256).astype(np.uint8))
    return grey

# subsample the image by a factor of 2
def ImageShrink2(img):
    retImg = skimage.measure.block_reduce(img, (2,2), np.mean)

    return retImg

def ComputeBitmaps(img):
    '''
    input img a gray-scale image
    return tb - threshold bitmap
           eb - exclusion bitmap
    '''

    med = np.median(img)
    tb = img > med
    eb = np.abs(img-med) >= 5
    return tb, eb

#shift bitmap by x0, y0
def BitmapShift(bm, x0, y0):
    shifted_x = np.full(bm.shape, False, dtype='bool')
    if x0 > 0:
        shifted_x[x0:] = bm[:-x0]
    elif x0 < 0:
        shifted_x[:x0] = bm[-x0:]
    else:
        shifted_x = bm

    shifted = np.full(shifted_x.shape, False, dtype='bool')
    if y0 > 0:
        shifted[:, y0:] = shifted_x[:, :-y0]
    elif y0 < 0:
        shifted[:, :y0] = shifted_x[:, -y0:]
    else:
        shifted = shifted_x
    

    return shifted


def getExpShift(img1, img2, shift_bits):
    if shift_bits > 0:
        sml_img1 = ImageShrink2(img1)
        sml_img2 = ImageShrink2(img1)
        cur_shift = getExpShift(sml_img1, sml_img2, shift_bits-1)
        del sml_img1, sml_img2
        cur_shift[0]*=2
        cur_shift[1]*=2
    else:
        cur_shift = [0, 0]
    
    tb1, eb1 = ComputeBitmaps(img1)
    tb2, eb2 = ComputeBitmaps(img2)

    min_err = img1.shape[0] * img1.shape[1]

    for i in range(-15, 15):
        for j in range(-15, 15):
            xs = cur_shift[0] + i
            ys = cur_shift[1] + j
            shifted_tb2 = BitmapShift(tb2, xs, ys)
            shifted_eb2 = BitmapShift(eb2, xs, ys)
            diff_b = np.logical_xor(tb1, shifted_tb2)
            diff_b = np.logical_and(diff_b, eb1)
            diff_b = np.logical_and(diff_b, shifted_eb2)
            err = np.sum(diff_b)

            if err < min_err:
                shift_ret = [xs, ys]
                min_err = err
    return shift_ret

def imgShift(img, x, y):
    shifted_x = np.full(img.shape, 0, dtype='uint8')
    if x > 0:
        shifted_x[x:] = img[:-x]
    elif x < 0:
        shifted_x[:x] = img[-x:]
    else:
        shifted_x = img
        
    shifted = np.full(img.shape, 0, dtype='uint8')
    if x > 0:
        shifted[x:] = img[:-x]
    elif x < 0:
        shifted[:x] = img[-x:]
    else:
        shifted = shifted_x

    return shifted


def align(image_list, level=2):
    print("===Start image alignment===")
    ref_indx=0
    print("reference image idx - {}".format(str(ref_indx)))
    ref = image_list[ref_indx]
    ret = []

    for i in range(0, len(image_list)):
        if i==ref_indx:
            ret.append(image_list[i])
            continue
        g_ref = rgb2gray(ref)
        g_t = rgb2gray(image_list[i])
        x, y = getExpShift(g_ref, g_t, level)

        ret.append(imgShift(image_list[i], x, y))
        print("img_idx: ", i,"delta_x: ", x, "delta_y: ", y)
        cv2.imwrite("align_{}.jpg".format(str(i)), (ret[0]*0.5+ret[-1]*0.5).astype(np.uint8))
    return ret
    