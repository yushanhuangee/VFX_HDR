import numpy as np
import math

def tonemap_global(hdr):
    sigma = 0.001
    a = 3
    Lw = 3

    
    #To deal with artifacts
    hdr_idx = hdr >1000
    hdr[hdr_idx] =1000

    mean = np.mean(np.log(hdr + sigma))
    L_w_av = math.exp(mean)
    L_m = a/L_w_av*hdr

    L_d = L_m/(1+L_m)*255
    L_d = L_d**2/255
    L_d_2 = L_m*(1+L_m/np.power(Lw,2))/(1+L_m)*255

    return L_d, L_d_2