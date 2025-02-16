def calculate_comp_metrics(image_1_path, image_2_path, save_path, colourmap_ssim=2, colourmap_rse=2):

    '''
    returns global ssim and rmse

    '''

    from skimage.metrics import structural_similarity
    from reslice_tif import reslice_image
    import cv2
    import numpy as np
    from scipy import stats

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

    filename_1 = image_1_path.split('\\')[-1].split('.')[0]
    filename_2 = image_2_path.split('\\')[-1].split('.')[0]
    img1 = reslice_image(image_1_path)[0]
    img2 = reslice_image(image_2_path)[0]
    
    (ssim, diff) = structural_similarity(img1, img2, full=True)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    imC = cv2.applyColorMap(diff, colourmap_codes[colourmap_ssim])
    cv2.imwrite(save_path+f'\\{filename_1}_to_{filename_2}_ssim.tiff', imC)
    
    
    RSE_map = np.sqrt(np.square(np.subtract(img1,img2)))
    
    RSE_map = cv2.normalize(RSE_map.astype('int32'), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    RSE_map = cv2.applyColorMap(RSE_map, colourmap_codes[colourmap_rse])
    cv2.imwrite(save_path+f'\\{filename_1}_to_{filename_2}_rse_map.tiff', RSE_map)
    
    RMSE = np.sqrt(np.square(np.subtract(img1,img2)).mean())
    
    #pearsoncorr = np.sum(img-img.mean()+ img2-img2.mean())/np.sqrt(np.sum(np.square(img-img.mean()))*np.sum(np.square(img2-img2.mean())))
    
    return ssim, RMSE
    
