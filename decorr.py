'''
This function is based on "Descloux, A., K. S. Grußmayer, and A. Radenovic. "Parameter-free image 
% resolution estimation based on decorrelation analysis."
% Nature methods (2019): 1-7." , where a detailed description of the method can be found

'''
import numpy as np
from scipy.signal import argrelextrema


def apodize(img):
    #TODO fix apodization so it only smooths out the edges
    window = np.outer(np.bartlett(img.shape[0]),np.bartlett(img.shape[1]))
    ap_array = img*window
    return ap_array


def coorcoef(I1, I2, c):
    np.seterr(divide='ignore', invalid='ignore')
    c1=c
    c2 = np.sqrt(np.sum(np.sum(I2*np.conj(I2))))
    cc = np.sum(np.sum(np.real(I1*np.conj(I2))))/((c1*c2))
    cc = (1000*cc)/1000
    return np.round(np.real(cc), 3)


def getDcorr(img,r=np.linspace(0,1,50),Ng=5):

    #input check
    if len(r) < 30:
        r = np.linspace(min(r),max(r),30)
    if Ng < 5:
        Ng = 5

    img = np.single(img)
    img = img[0:img.shape[0]- (1-(img.shape[0]%2)), 0:img.shape[1]- (1-(img.shape[1]%2))]
    X, Y = np.meshgrid(np.linspace(-1,1,np.shape(img)[1]),np.linspace(-1,1,np.shape(img)[0]))
    R = np.sqrt(X*X + Y*Y)
    Nr = len(r)
    #In : Fourier normalized image
    In = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(img)))
    In = In/abs(In)
    In = np.where(np.isinf(In), 0, In)
    In = np.where(np.isnan(In), 0, In)
    mask0 = np.where(R*R < 1, 1, 0)
    In = mask0*In # restric all the analysis to the region r < 1
    Ik = mask0*np.fft.fftshift(np.fft.fftn(np.fft.fftshift(img)))
    d0 = []
    c = np.sqrt(np.sum(np.sum(Ik*np.conj(Ik))))
    r0 = np.linspace(r[0],r[-1],Nr)
    for k in range(len(r0)-1, -1, -1):
        cc = coorcoef(Ik,(np.where(R*R < r0[k]**2, 1, 0))*In,c)
        if np.isnan(cc):
            cc = 0
        d0.append(cc) 
    d0.reverse()
    
    #plt.plot(r0, d0)
    d0=np.array(d0)
    
    ind0 = argrelextrema(d0, np.greater)[0]
    if len(ind0) >=1:
        snr0 = d0[ind0[0]]
        k0 = r[ind0[0]]
        gMax = 2/r0[ind0[0]]
    else:
        snr0 = 0
        k0 = r[-1]
        gMax = max(np.shape(img)[0],np.shape(img)[1])/2
    if np.isinf(gMax):
        gMax = max(np.shape(img)[0],np.shape(img)[1])/2
    
    #search of highest frequency peak
    g = [np.shape(img)[0]/4]
    g.extend(np.exp(np.linspace(np.log(gMax),np.log(0.15),Ng)))
    d = np.zeros((Nr,2*Ng+1))
    
    kc = []
    kc.append(k0)
    SNR = []
    SNR.append(snr0)
    ind0 = 1
    
    for ref in range(2):
        for h in range(1, len(g)):
            
            Ir = Ik*(1 - np.exp(-2*g[h]*g[h]*R*R)) 
            c = np.sqrt(np.sum(np.sum(Ir*np.conj(Ir))))
            for k in range(len(r)-1, ind0-1, -1):
                cc = coorcoef((np.where(R*R < r[k]**2, 1, 0))*Ir,(np.where(R*R < r[k]**2, 1, 0))*In,c)
                if np.isnan(cc):
                    cc = 0
                d[k, h + Ng*ref] = cc
                ind = argrelextrema(d[k:, h + Ng*ref], np.greater)[0]
                
                if len(ind) >= 1:
                    snr = d[ind[0],h + Ng*ref]
                    kc.append(r[ind[0]+1])
                    SNR.append(snr)
                    ind[0] = ind[0] +k-1
                else:
                    ind = ind +k-1
                  
            #plt.plot(r0, d[:, h + Ng*ref])
            if kc:
                kcMax, ind = np.max(kc), np.argmax(kc)
                A0 = snr0 
            else:
                kcMax = r[2]
                A0 = 0
    
    return kcMax, np.real(A0)
