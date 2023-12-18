# Edited from https://github.com/HzFu/EyeQ/blob/master/EyeQ_preprocess/EyeQ_process_main.py
from absl import logging
from modules.dataset.preprocess_dataset.eyeq import fundus_prep as prep
from modules.utils import (ProgressBar)
import glob
import os
import cv2 as cv
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def eyeq_process(dataset_part, image_list, source_dir, destination_dir):
    """Preprocess images of EyeQ dataset from EyePACS dataset"""
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        logging.info('\'{}\' directory is created!'.format(destination_dir))

    total_src_img_num = len(image_list)
    total_prep_img_num = 0
    existed_prep_img_num = 0
    completed_new_prep_img_num = 0
    failed_new_prep_img_num = 0
    prog_bar = ProgressBar(total_src_img_num, total_prep_img_num)
    
    for image in image_list:
        total_prep_img_num += 1
        prog_bar.update('Preprocessing \'{}\' images: {}/{}'.format(dataset_part, total_prep_img_num, total_src_img_num))

        img_name = image['image']
        src_img_path = os.path.join(source_dir, img_name)
        dest_img_path = os.path.join(destination_dir, img_name)

        if os.path.exists(dest_img_path):
            existed_prep_img_num += 1
            logging.info('\'{}\' image already existed!!! Skip preprocessing of this image'.format(dest_img_path))
            continue
        
        try:
            img = prep.imread(src_img_path)
            r_img, borders, mask = prep.process_without_gb(img)
            r_img = cv.resize(r_img, (800, 800))
            prep.imwrite(dest_img_path, r_img)
            # mask = cv.resize(mask, (800, 800))
            # prep.imwrite(os.path.join(destination_dir, 'mask', img_name), mask)
            completed_new_prep_img_num += 1
        except:
            failed_new_prep_img_num += 1
            logging.info('\'{}\' image has error during preprocessing!!!'.format(src_img_path))
            continue
    
    return exist_prep_img_num, completed_new_prep_img_num, failed_new_prep_img_num