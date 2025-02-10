def fft2_calc(image_name, colourmap=cv2.COLORMAP_JET):

    import numpy as np
    from libtiff import TIFF
    import skimage
    import cv2


    tif = TIFF.open(image_name+'.tif', mode='r')
    for i in tif.iter_images():
        fourier = cv2.dft(np.float32(i), flags=cv2.DFT_COMPLEX_OUTPUT)
        fourier_shift = np.fft.fftshift(fourier)
        magnitude = 20*np.log(cv2.magnitude(fourier_shift[:,:,0],fourier_shift[:,:,1]))
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        imC = cv2.applyColorMap(magnitude, colourmap)
        cv2.imwrite('fourier.tiff', imC)
    tif.close()
    '''
  cv::COLORMAP_AUTUMN = 0 ,
  cv::COLORMAP_BONE = 1 ,
  cv::COLORMAP_JET = 2 ,
  cv::COLORMAP_WINTER = 3 ,
  cv::COLORMAP_RAINBOW = 4 ,
  cv::COLORMAP_OCEAN = 5 ,
  cv::COLORMAP_SUMMER = 6 ,
  cv::COLORMAP_SPRING = 7 ,
  cv::COLORMAP_COOL = 8 ,
  cv::COLORMAP_HSV = 9 ,
  cv::COLORMAP_PINK = 10 ,
  cv::COLORMAP_HOT = 11 ,
  cv::COLORMAP_PARULA = 12 ,
  cv::COLORMAP_MAGMA = 13 ,
  cv::COLORMAP_INFERNO = 14 ,
  cv::COLORMAP_PLASMA = 15 ,
  cv::COLORMAP_VIRIDIS = 16 ,
  cv::COLORMAP_CIVIDIS = 17 ,
  cv::COLORMAP_TWILIGHT = 18 ,
  cv::COLORMAP_TWILIGHT_SHIFTED = 19 ,
  cv::COLORMAP_TURBO = 20 ,
  cv::COLORMAP_DEEPGREEN = 21 
    '''


 


