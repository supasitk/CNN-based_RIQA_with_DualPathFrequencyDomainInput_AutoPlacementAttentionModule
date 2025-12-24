from absl import logging
import os
import fnmatch

from modules.utils import (ProgressBar)
from modules.dataset.additional_preprocess_dataset.eyeq.addPrepImg_functions import (gen_addPrepImg_funtion)


def eyeq_additional_preprocess(input_root_dir, list_input_subdir, output_root_dir, addPrep_method, addPrep_method_configs):
    """Additional preprocessing image after doing the preprocessed images for EyeQ from the original images of EyePACS dataset"""
    # generate function of additional preprocessing images
    addPrepImg_fn = gen_addPrepImg_funtion(addPrep_method=addPrep_method, 
                                           addPrep_method_configs=addPrep_method_configs)

    for input_subdir in list_input_subdir:
        input_dir = os.path.join(input_root_dir, input_subdir)
        
        output_dir = os.path.join(output_root_dir, input_subdir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info('\'{}\' directory is created!'.format(output_dir))

        list_name_of_input_imgs = fnmatch.filter(os.listdir(input_dir), '*.jpeg')
        total_num_of_input_imgs = len(list_name_of_input_imgs)
        total_num_of_addPrepImgs = 0
        num_of_existed_addPrepImgs = 0
        num_of_completed_new_addPrepImgs = 0
        num_of_failed_new_addPrepImgs = 0
        prog_bar = ProgressBar(total_num_of_input_imgs, total_num_of_addPrepImgs)

        for img_name in list_name_of_input_imgs:
            total_num_of_addPrepImgs += 1
            prog_bar.update('Additional preprocessing \'{}\'/\'{}\' images: {}/{}'.format(addPrep_method, input_subdir, total_num_of_addPrepImgs, total_num_of_input_imgs))

            input_img_path = os.path.join(input_dir, img_name)
            output_img_path = os.path.join(output_dir, img_name)

            if os.path.exists(output_img_path):
                num_of_existed_addPrepImgs += 1
                logging.info('\'{}\' image already existed!!! Skip additional preprocessing of this image'.format(output_img_path))
                continue

            try:
                addPrepImg_fn.apply(input_img_path, output_img_path)
                num_of_completed_new_addPrepImgs += 1
            except:
                num_of_failed_new_addPrepImgs += 1
                logging.info('\'{}\' image has error during additional preprocessing \'{}\'/\'{}\' !!!'.format(input_img_path, addPrep_method, input_subdir)) 
                continue
        
        logging.info('{}, {}, {} images of existed, completed, failed additional preprocessing for method \'{}\'/\'{}\'.'.format(num_of_existed_addPrepImgs, num_of_completed_new_addPrepImgs, num_of_failed_new_addPrepImgs, addPrep_method, input_subdir))
