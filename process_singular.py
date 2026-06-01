import numpy as np
from reslice_tif import reslice_image
import fft2
import tifffile as tiff
from decorr import apodize, getDcorr
import fwhm

def calculate_metrics(path, pixel_size, output_path, colourmap=21):

    filename = path.split('\\')[-1]
    filename = filename.split('.')[0]
    imgarr = tiff.imread(path)
    
    
    fft2.fft2_calc(imgarr, filename, output_path, colourmap)
    img = apodize(imgarr)
    kcmax, a0 = getDcorr(img,r=np.linspace(0,1,50),Ng=10)
    ndcont, std = fwhm.calculate_fwhm(imgarr)

    return 2*pixel_size/kcmax, a0, std
            

