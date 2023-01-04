import exifread
import glob
import cv2
import numpy as np
import math
import align
#import matplotlib.pyplot as plt
import tonemap


def w(z):
    sigma = 1e-4
    if z <= 127:
        return z+sigma
    else:
        return 255-z + sigma

def getG(z, time, select, _lambda):
    pixels = []
    for pixel in select:
        tmp = []
        for img in z:
            tmp.append(img[pixel[1], pixel[0]])
        pixels.append(tmp)
    pixels = np.asarray(pixels)

    matrix_a = []
    matrix_b = []
    N = len(pixels)
    P = len(z)
    for i in range(N):
        for j in range(P):
            z_ij = pixels[i][j]
            tmp_0 = np.zeros(256 + N)
            tmp_0[z_ij] = w(z_ij)
            tmp_0[256 + i] = -w(z_ij)
            matrix_a.append(tmp_0)
            matrix_b.append(w(z_ij) * np.log(time[j]))

    tmp_0 = np.zeros(256 + N)
    tmp_0[127] = 1
    matrix_a.append(tmp_0)
    matrix_b.extend([0] * 255)

    for i in range (254):
        tmp_0 = np.zeros(256 + N)
        tmp_0[i] = _lambda * w(i+1)
        tmp_0[i+1] = (-2) * _lambda * w(i+1)
        tmp_0[i+2] = _lambda * w(i+1)
        matrix_a.append(tmp_0)

    # Ax = b => x = (A^-1)Ax = (A^-1)b
    pseudoInv = np.linalg.pinv(matrix_a)
    matrix_x = np.dot(pseudoInv, matrix_b)

    return np.asarray(matrix_x[0:256])

def getE(z, time, g):
    img_x, img_y = z.shape[1:3]
    P = len(z)
    upper = np.zeros((img_x, img_y), 'float32')
    lower = np.zeros((img_x, img_y), 'float32')

    for j in range(P):
        print("image "+ str(j))
        for x in range(img_x):
            for y in range(img_y):
                # z_ij = z[j][x][y]
                upper[x][y] += w(z[j][x][y]) * (g[z[j][x][y]] - np.log(time[j]))
                lower[x][y] += w(z[j][x][y])


    Ei = np.zeros((img_x, img_y), 'float32')
    
    for x in range(img_x):
        for y in range(img_y):
            if(lower[x][y] == 0):
                Ei[x][y] = 1
            else:
                Ei[x][y] = np.exp(upper[x][y] / lower[x][y])


    return Ei 


if __name__=='__main__':
    #get original image and informations
    
    original_data_paths = glob.glob('../data/original/*.JPG')
    original_data_paths.sort()
    images = []
    z = [[], [], []]
    exp_times = []

    for path in original_data_paths:
        f = open(path, 'rb')

        # Return Exif tags
        tags = exifread.process_file(f)
        exposure_time = float(tags['EXIF ExposureTime'].values[0])
        exp_times.append(exposure_time)
        im = cv2.imread(path) #(H, W, 3)
        im = cv2.resize(im, (1800, 1200), interpolation = cv2.INTER_AREA)
        cv2.imwrite("input_compress.jpg", im)
        images.append(im)


    #random sample select
    x = np.random.randint(20, images[0].shape[1]-20, size=150)
    y = np.random.randint(20, images[0].shape[0]-20, size=150)
    select = np.stack((x, y), axis=1)
    
    #image alignment
    images = align.align(images)
    input_im = images[0]

    #fig, ax=plt.subplots()
    #ax.imshow(input_im)
    #manually_select = np.asarray(plt.ginput(10, timeout=-1)).astype(np.uint8)
    manually_select = np.asarray([[175,131],[255,46], [184,182],[151,5],[127,240],[76,129],[92,69],[229,202],[121,142],[67,213]])
    select = np.concatenate([select, manually_select], axis=0)


    for px, py in select:
        input_im[py, px, :] = np.asarray([[256,0,0]])
    cv2.imwrite("selected_point.jpg", input_im)
    

    for im in images:
        z[0].append(im[:,:,0])
        z[1].append(im[:,:,1])
        z[2].append(im[:,:,2])
       
    
    z = np.array(z)

        

    #get g curve for each dimension
    g = []
    _lambda = 500
    for i in range(3):
        g.append(getG(z[i], exp_times, select, _lambda))
    g = np.array(g)

    #calculate E

    E = np.zeros((len(images[0]), len(images[0][0]), 3), 'float32')
    for i in range(3):
        print("Calculating channel {} of E".format(str(i)))
        E[:, :, i]= getE(z[i], exp_times, g[i])
    print(E.shape)

    cv2.imwrite("result.hdr", E)
    
    E = cv2.imread("result.hdr", flags=cv2.IMREAD_ANYDEPTH)

    ldr, ldr2 = tonemap.tonemap_global(E)
    #cv2.imwrite("result_tonemapped.jpg",ldr2)
    cv2.imwrite("result_tonemapped.jpg",ldr)

    cv2tonemapper = cv2.createTonemap(2.2)
    ldr_cv2 = cv2tonemapper.process(E)*255
    cv2.imwrite("tone_bi.jpg", ldr_cv2)

        
        
