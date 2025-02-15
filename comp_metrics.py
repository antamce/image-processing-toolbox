def calculate_comp_metrics(image_1_path, image_2_path, colourmap_ssim=2, colourmap_rse=2, save_path=''):

    '''
    returns global ssim and rmse

    '''

    from skimage.metrics import structural_similarity
    from libtiff import TIFF
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

    tif = TIFF.open(image_1_path, mode='r')
    tif2 = TIFF.open(image_2_path, mode='r')
    for i in tif.iter_images():
        img = np.array(i)
    for j in tif2.iter_images():
        img2 = np.array(j)
    
    (score, diff) = structural_similarity(img, img2, full=True)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    imC = cv2.applyColorMap(diff, colourmap_codes[colourmap_ssim])
    cv2.imwrite(save_path+'\\ssim.tiff', imC)
    
    
    RSE_map = np.sqrt(np.square(np.subtract(img,img2)))
    
    RSE_map = cv2.normalize(RSE_map.astype('int32'), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    RSE_map = cv2.applyColorMap(RSE_map, colourmap_codes[colourmap_rse])
    cv2.imwrite(save_path+'\\rse_map.tiff', RSE_map)

    RMSE = np.sqrt(np.square(np.subtract(img,img2)).mean())
    print(np.subtract(img,img2).mean())
    #pearsoncorr = np.sum(img-img.mean()+ img2-img2.mean())/np.sqrt(np.sum(np.square(img-img.mean()))*np.sum(np.square(img2-img2.mean())))
    
    tif.close()
    tif2.close()
    return score, RMSE
    
