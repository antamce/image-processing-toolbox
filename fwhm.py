from imutils import contours
from skimage import measure
from skimage import draw
import imutils
import numpy as np
import cv2
from scipy import signal
from scipy.stats import norm


def calculate_fwhm(img, gaussian_blur=5, binarization_threshold=0.7, erode_iterations=2, dilate_iterations=3, min_pixel_count_mask=80, resampling_rate=15):
    
    contours_array = []
    blurred = cv2.GaussianBlur(img, (gaussian_blur, gaussian_blur), 0)
    thresh = cv2.threshold(blurred, blurred.max()*binarization_threshold, blurred.max(), cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=erode_iterations)
    thresh = cv2.dilate(thresh, None, iterations=dilate_iterations)
    thresh = np.where(thresh > 0, 1, 0)
    labels = measure.label(thresh, connectivity=2)

    mask = np.zeros(thresh.shape, dtype="uint8")
    for label in np.unique(labels):
        if label == 0:
            continue
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        if numPixels > min_pixel_count_mask:
            mask = cv2.add(mask, labelMask)
                
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if bool(cnts):
        cnts = contours.sort_contours(cnts)[0]
        (x, y, w, h) = cv2.boundingRect(cnts[0])
        lineim = cv2.line(img.copy(), (x, y+h-1), (x+w-1, y), (255, 255, 255), 2)
        line = np.transpose(np.array(draw.line(x, y+h-1, x+w-1, y)))
        data = img[line[:, 1], line[:, 0]]
        contours_array.append(signal.resample((data), resampling_rate))
        ndcont = np.array(contours_array)
        ndcont = np.round(ndcont, 10)
        mean,std=norm.fit(ndcont)
    else:
        ndcont, std = 0, 0
    return ndcont, std