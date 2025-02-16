def fft2_calc(image_arr, filename, output_path, colourmap=2):

    import numpy as np
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


    
    fourier = cv2.dft(np.float32(image_arr), flags=cv2.DFT_COMPLEX_OUTPUT)
    fourier_shift = np.fft.fftshift(fourier)
    magnitude = 20*np.log(cv2.magnitude(fourier_shift[:,:,0],fourier_shift[:,:,1]))
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    imC = cv2.applyColorMap(magnitude, colourmap_codes[colourmap])
    cv2.imwrite(f'{output_path}\\{filename}_fourier.tiff', imC)
    



 


