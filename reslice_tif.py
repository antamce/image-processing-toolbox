import numpy as np
import tifffile as tiff


def reslice_image(path, save=False, output_path=''):
    
    tif = tiff.imread(path)
 
    xz_reslice = tif.transpose(1, 0, 2)
    yz_reslice = tif.transpose(2, 0, 1)
    

    if save:


        filename = path.split('\\')[-1]
        filename = filename.split('.')[0]
						
        with tiff.TiffWriter(f"{output_path}\\{filename}_xz.tif") as stack_1:
            stack_1.write(xz_reslice, contiguous=True)

        with tiff.TiffWriter(f"{output_path}\\{filename}_yz.tif") as stack_2:
            stack_2.write(yz_reslice, contiguous=True)
        
        with tiff.TiffWriter(f"{output_path}\\{filename}_xy.tif") as stack_3:
            stack_3.write(tif, contiguous=True)

    
    return tif, xz_reslice, yz_reslice
    
