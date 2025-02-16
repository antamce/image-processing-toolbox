import numpy as np
from reslice_tif import reslice_image
import fft2
from decorr import apodize, getDcorr
import fwhm

def calculate_metrics(path, pixel_size, output_path, colourmap=21):

    filename = path.split('\\')[-1]
    filename = filename.split('.')[0]
    reslice_arr = reslice_image(path)[0]
    
    #TODO think about apodization here
    fft2.fft2_calc(reslice_arr[0], filename, output_path, colourmap)
    img = apodize(reslice_arr[0])
    kcmax, a0 = getDcorr(img,r=np.linspace(0,1,50),Ng=10)
    ndcont, std = fwhm.calculate_fwhm(reslice_arr[0])

    return 2*pixel_size/kcmax, a0, std
            

