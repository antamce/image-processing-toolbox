# Image processing toolbox

A set of scripts to calculate:
1. Resolution 
2. SNR
3. Mean gaussian-fitted FWHM of objects in the picture
4. FFT
5. SSIM, RMSE between two images

 of .tiff images or stacks. 

 For stacks, resolution and SNR are calculated in all three planes (XY, YZ, XZ). 
 
 The resolution and SNR are calculated based on the algorithm described in here : "Descloux, A., K. S. Grußmayer,and A. Radenovic. "Parameter-free image resolution estimation based on decorrelation analysis. Nature methods (2019): 1-7."

If you use any of these scripts for your research, please cite these papers:

1. Sachuk A, Volkova E, Rakovskaya A, Chukanov V, Pchitskaya E. NeuroDecon: A Neural Network-Based Method for Three-Dimensional Deconvolution of Fluorescent Microscopic Images. International Journal of Molecular Sciences. 2025; 26(18):8770. https://doi.org/10.3390/ijms26188770

2. Descloux, A., Grußmayer, K.S. & Radenovic, A. Parameter-free image resolution estimation based on decorrelation analysis. Nat Methods 16, 918–924 (2019). https://doi.org/10.1038/s41592-019-0515-7
