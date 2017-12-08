# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 17:27:49 2017

@author: Bami Van Hohenheim
"""

# %% import library

import time as t
import numpy as np
import imageio as imo
import math as m
from PIL import Image as im
import multiprocessing as mp

# %% read image

def read_image(file):
    img=imo.imread(file)
    size=np.shape(img)
    
    return img,size


# %% make gaussian kernel

def make_gaussian_kernel(sigma,size):
    denum=1/(2*np.pi*np.power(sigma,2))
    dexp=2*m.pow(sigma,2)
    kernel=[[0 for x in range(size)] for y in range(size)] 
    
    for x in range(0, size):
        for y in range(0, size):
            xyval=(m.pow(x-2,2)+m.pow(y-2,2))
            befexp=(xyval/dexp)*(-1)
            exp=m.exp(befexp)
            val=denum*exp
            kernel[x][y]=val    

    raiseval=kernel[2][2]/kernel[0][0]           
    sumkernel=0
    
    for x in range(0, size):
        for y in range(0, size):
            kernel[x][y]=np.around(kernel[x][y]*raiseval)
            sumkernel+=round(kernel[x][y],0)
    
            
    return kernel,sumkernel+1         

# %% make blur image
    
def smoothing(img,kernel,kernsum,time):
    row=len(img)
    col=len(img[0])
    blur=img
    
    for tm in range(time):
        for x in range(0,row):
            for y in range(0,col):
                if x==0 or y==0 or x==row-1 or y==col-1:
    #                print("do not change the value, one of them is upper bound or lower bound")
                    blur[x][y]=img[x][y]
                elif x==1 or y==1 or x==row-2 or y==col-2:
    #                print("do not change the value, one of them is 1 step before upper bound or 1 step after lower bound")
                    blur[x][y]=img[x][y]
                else:
    #                print("you can change the value, congratulations")
                    blur[x][y]=(
                        img[x-2][y-2]*kernel[0][0]+img[x-2][y-1]*kernel[0][1]+img[x-2][y]*kernel[0][2]
                        +img[x-2][y+1]*kernel[0][3]+img[x-2][y+2]*kernel[0][4]
                        +img[x-1][y-2]*kernel[1][0]+img[x-1][y-1]*kernel[1][1]+img[x-1][y]*kernel[1][2]
                        +img[x-1][y+1]*kernel[1][3]+img[x-1][y+2]*kernel[1][4]
                        +img[x][y-2]*kernel[2][0]+img[x][y-1]*kernel[2][1]+img[x][y]*kernel[2][2]
                        +img[x][y+1]*kernel[2][3]+img[x][y+2]*kernel[2][4]
                        +img[x+1][y]*kernel[3][0]+img[x+1][y-1]*kernel[3][1]+img[x+1][y]*kernel[3][2]
                        +img[x+1][y+1]*kernel[3][3]+img[x+1][y+2]*kernel[3][4]
                        +img[x+2][y-2]*kernel[4][0]+img[x+2][y-1]*kernel[4][1]+img[x+2][y]*kernel[4][2]
                        +img[x+2][y+1]*kernel[4][3]+img[x+2][y+2]*kernel[4][4]
                        )/kernsum
        print('Smoothing done in:',tm+1,'time')
        
    return blur
    
# %% main program
            
if __name__ == "__main__":
    start = t.clock()
    
    filename='ganteng.jpg'
    imgori,ori_size=read_image(filename)
    with mp.Pool(4) as p:
        gau=p.starmap(make_gaussian_kernel,[(1,5)])

    gauker,gausum=gau[0][0],gau[0][1]
    print("Kernel:\n",gauker)
    print("\nKernel sum:",gausum)

    times=2
    with mp.Pool(4) as p:
        blur=p.starmap(smoothing,[(imgori,gauker,gausum,times)])
    
    imgblur=blur[0]
    save=im.fromarray(imgblur,mode=None)
    save.save("BlurParalel.png","PNG")
    
    print("--- Gaussian Blur done in {0:.5f} minutes ---".format((t.clock() - start)/60))