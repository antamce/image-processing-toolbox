import os
from pathlib import Path
from process_stacks import calculate_stack_metrics, calculate_fwhm_stack
from process_singular import calculate_metrics
from comp_metrics import calculate_comp_metrics



def analyze_tiff_stack(dir_path, group, pixel_size, z_step, output_path=r'output_folder'):
    '''
    dir_path: directory path to the .tiffs
    group: regex for all images in the group
    pixel_size: the resolution and fwhm will be scaled accordingly
    z_step: make sure the units match pixel size

    '''
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    pathlist = Path(dir_path).glob(f'*{group}*.tif')

    for path in pathlist:
        path_in_str = str(path)
        #saves metrics and fwhm into .xlsx files
        #functions return mean values if needed
        calculate_stack_metrics(path_in_str, pixel_size, z_step, output_path)
        calculate_fwhm_stack(path_in_str, pixel_size, z_step, output_path)
    

def analyze_tiff(dir_path, group, pixel_size, output_path=r'output_folder',  colourmap=21):
    '''
    dir_path: directory path to the .tiffs
    group: regex for all images in the group
    pixel_size: the resolution and fwhm will be scaled accordingly
    

    '''
    if not os.path.exists(output_path):
            os.makedirs(output_path)
    pathlist = Path(dir_path).glob(f'*{group}*.tif')
    for path in pathlist:
        path_in_str = str(path)
        #TODO put into an array
        resolution, SNR, fwhm = calculate_metrics(path_in_str, pixel_size, output_path, colourmap)
    return resolution, SNR, fwhm

def compare_tiffs(image_1_path, image_2_path, output_path=r'output_folder', colourmap_ssim=2, colourmap_rse=2):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    ssim, RMSE = calculate_comp_metrics(image_1_path, image_2_path, output_path,  colourmap_ssim, colourmap_rse)
    return ssim, RMSE
     
