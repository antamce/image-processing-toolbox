def reslice_image(image_name):
    
    import numpy as np
    from libtiff import TIFF
    import skimage
    import os


    reslice_arr = []
    tif = TIFF.open(image_name+'.tif', mode='r')

    for i in tif.iter_images():
        reslice_arr.append(i)

    imarray = np.array(reslice_arr)

    xz_reslice = imarray.transpose(1, 0, 2)
    yz_reslice = imarray.transpose(2, 0, 1)

    skimage.io.imsave(f"{image_name}_xz.tif", xz_reslice)
    skimage.io.imsave(f"{image_name}_yz.tif", yz_reslice)
    tif.close()
    os.replace(f"{image_name}.tif", f"{image_name}_xy.tif")
    
