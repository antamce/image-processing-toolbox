def fft2_calc(image_name, colourmap=2):

    import numpy as np
    from libtiff import TIFF
    import skimage
    import cv2

    colourmap_codes = { 0 : cv2.COLORMAP_AUTUMN,
                        1 : cv2.COLORMAP_BONE,
                        2 : cv2.COLORMAP_JET,
                        3 : cv2.COLORMAP_WINTER,
                        4 : cv2.COLORMAP_RAINBOW,
                        5 : cv2.COLORMAP_OCEAN,
                        6 : cv2.COLORMAP_SUMMER,
                        7 : cv2.COLORMAP_SPRING,
                        8 : cv2.COLORMAP_COOL,
                        9 : cv2.COLORMAP_HSV,
                        10 : cv2.COLORMAP_PINK,
                        11 : cv2.COLORMAP_HOT,
                        12  : cv2.COLORMAP_PARULA,
                        13 : cv2.COLORMAP_MAGMA,
                        14 : cv2.COLORMAP_INFERNO,
                        15 : cv2.COLORMAP_PLASMA,
                        16 : cv2.COLORMAP_VIRIDIS,
                        17 : cv2.COLORMAP_CIVIDIS,
                        18 : cv2.COLORMAP_TWILIGHT,
                        19 : cv2.COLORMAP_TWILIGHT_SHIFTED,
                        20 : cv2.COLORMAP_TURBO,
                        21 : cv2.COLORMAP_DEEPGREEN}


    tif = TIFF.open(image_name, mode='r')
    for i in tif.iter_images():
        fourier = cv2.dft(np.float32(i), flags=cv2.DFT_COMPLEX_OUTPUT)
        fourier_shift = np.fft.fftshift(fourier)
        magnitude = 20*np.log(cv2.magnitude(fourier_shift[:,:,0],fourier_shift[:,:,1]))
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        imC = cv2.applyColorMap(magnitude, colourmap_codes[colourmap])
        cv2.imwrite('fourier.tiff', imC)
    tif.close()



 


