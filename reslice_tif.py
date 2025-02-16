import numpy as np
from libtiff import TIFF

def reslice_image(path, save=False, output_path=''):
    
    tif = TIFF.open(path, mode='r')
    reslice_arr = []
    for i in tif.iter_images():
        reslice_arr.append(i)

    imarray = np.array(reslice_arr)
    xz_reslice = imarray.transpose(1, 0, 2)
    yz_reslice = imarray.transpose(2, 0, 1)
    tif.close()

    if save:

        import skimage
        import os

        filename = path.split('\\')[-1]
        filename = filename.split('.')[0]
        skimage.io.imsave(f"'{output_path}\\{filename}_xz.tif", xz_reslice)
        skimage.io.imsave(f"'{output_path}\\{filename}_yz.tif", yz_reslice)
        os.replace(f"'{output_path}\\{filename}.tif", f"{filename}_xy.tif")
    
    return imarray, xz_reslice, yz_reslice
    
