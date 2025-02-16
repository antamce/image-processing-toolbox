import numpy as np
from libtiff import TIFF
from reslice_tif import reslice_image
import pandas as pd
from decorr import apodize, getDcorr
import fwhm

def calculate_stack_metrics(path, pixel_size, z_step, output_path):
    '''
    Returns mean values for resolution, decorrelatoin SNR
    for image stacks for xy, xz, yz planes
    
     '''
    filename = path.split('\\')[-1]
    filename = filename.split('.')[0]
    reslice_arrs = reslice_image(path) #xy, xz, yz
    resolution_arr = np.zeros(((max(np.shape(reslice_arrs[1])[0], np.shape(reslice_arrs[2])[0])), 6))
    
    for n, i in enumerate(reslice_arrs):
        for m, j in enumerate(i):
            img = apodize(j)
            Nr = 50 #fourier space sampling
            Ng = 10 #number of inner refinement iterations, >= 5
            kcmax, a0 = getDcorr(img,r=np.linspace(0,1,Nr),Ng=Ng )
            if n==0:
                resolution_arr[m, n] = (2*pixel_size/kcmax)
            else:
                resolution_arr[m, n] = (2*z_step/kcmax)
            resolution_arr[m, n+3] = a0

    #TODO um well at least something
    df = pd.DataFrame(resolution_arr)
    df.columns = ['Resolution XY', 'Resolution XZ', 'Resolution YZ', 'SNR XY', 'SNR XZ', 'SNR YZ']
    df.replace(to_replace=0, value=np.NaN, inplace=True)
    df['Resolution XY'].dropna(inplace=True)
    df['Resolution XZ'].dropna(inplace=True)
    df['Resolution YZ'].dropna(inplace=True)
    df['SNR XY'].dropna(inplace=True)
    df['SNR XZ'].dropna(inplace=True)
    df['SNR YZ'].dropna(inplace=True)
    df.to_excel(f'{output_path}\\{filename}_dcorr_output.xlsx')
    
    return df['Resolution XY'].mean(), df['Resolution XZ'].mean(), df['Resolution YZ'].mean(), df['SNR XY'].mean(), df['SNR XZ'].mean(), df['SNR YZ'].mean()

def calculate_fwhm_stack(path, pixel_size, z_step, output_path):
    '''
    Returns mean gaussian-fitted fwhm
    for image stacks for xy, xz, yz planes
     
     '''
    filename = path.split('\\')[-1]
    filename = filename.split('.')[0]
    reslice_arrs = reslice_image(path) #xy, xz, yz
    fwhm_arr = np.zeros(((max(np.shape(reslice_arrs[1])[0], np.shape(reslice_arrs[2])[0])), 3))
    
    for n, i in enumerate(reslice_arrs):
        for m, j in enumerate(i):
            ndcont, std = fwhm.calculate_fwhm(j)
            if n==0:
                fwhm_arr[m, n] = (pixel_size*std)
            else:
                fwhm_arr[m, n] = (z_step*std)
    
    df = pd.DataFrame(fwhm_arr)
    df.replace(to_replace=0, value=np.NaN, inplace=True)
    df.columns = ['xy', 'xz', 'yz']
    df['xy'].dropna(inplace=True)
    df['xz'].dropna(inplace=True)
    df['yz'].dropna(inplace=True)
    df.to_excel(f'{output_path}\\{filename}_fwhm_output.xlsx')
    
    return df['xy'].mean(), df['xz'].mean(), df['yz'].mean()
