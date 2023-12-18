from absl import logging
import os
import fnmatch
import tensorflow as tf

from modules.utils import (read_csv_to_list_of_dict)
from modules.dataset.preprocess_dataset.eyeq.EyeQ_process_main import (eyeq_process)

def gen_dataset(dataset_name, dataset_cfg):
    """Generate train/validation/test dataloader of the dataset"""

    if dataset_name == 'EyeQ': # Eye Quality (EyeQ) dataset
        # check if the preprocessed images of the dataset existed
        # if not existing, the preprocessed images firstly need to be created
        ds_root_dir = dataset_cfg['dataset_root_dir']
        prep_ds_root_dir = os.path.join(ds_root_dir, dataset_cfg['dataset_preprocessed_dir_name'])
        prep_ds_train_subdir = os.path.join(prep_ds_root_dir, dataset_cfg['dataset_preprocessed_train_dir_name'])
        prep_ds_val_subdir = os.path.join(prep_ds_root_dir, dataset_cfg['dataset_preprocessed_val_dir_name'])
        prep_ds_test_subdir = os.path.join(prep_ds_root_dir, dataset_cfg['dataset_preprocessed_test_dir_name']) 

        val_from_train = dataset_cfg['val_from_train']
        val_portion = dataset_cfg['val_portion']

        if not os.path.exists(prep_ds_root_dir): # case of the preprocessed images of the dataset not existed
            logging.info('The preprocessed images of dataset cannot be found. Starting to create....') 
            
            # Read list of train/val/test images for EyeQ which is a subset of all train/val/test images of EyePACS dataset
            # these lists get from the original source code of EyeQ dataset project
            # the csv file for train/val/test image list of EyeQ dataset has [id, image, quality, DR_grade] columns 
            # the format of train/val/test images list is a list of dictionaries
            ds_src_dir = os.path.join(ds_root_dir, dataset_cfg['dataset_src_dir_name'])
            ds_src_train_val_test_list_dir = os.path.join(ds_src_dir, dataset_cfg['dataset_src_train_val_test_list_dir_name'])
            train_img_list_csv = os.path.join(ds_src_train_val_test_list_dir, dataset_cfg['dataset_src_train_list_filename'])
            val_img_list_csv = os.path.join(ds_src_train_val_test_list_dir, dataset_cfg['dataset_src_val_list_filename'])
            test_img_list_csv = os.path.join(ds_src_train_val_test_list_dir, dataset_cfg['dataset_src_test_list_filename'])

            train_img_list = read_csv_to_list_of_dict(train_img_list_csv, ',')
            val_img_list = list()
            if val_from_train == False:
                val_img_list = read_csv_to_list_of_dict(val_img_list_csv, ',')    
            test_img_list = read_csv_to_list_of_dict(test_img_list_csv, ',')
            logging.info('Found the number of images for train, validation, test: {}, {}, {}'.format(len(train_img_list), len(val_img_list), len(test_img_list)))

            # Preprocess images of EyeQ dataset from EyePACS dataset
            org_ds_root_dir = dataset_cfg['original_dataset_root_dir']
            train_val_test = ['train', 'validation', 'test']
            img_lists = [train_img_list, val_img_list, test_img_list]
            src_dirs = [os.path.join(org_ds_root_dir, dataset_cfg['original_dataset_train_dir_name']), 
                        os.path.join(org_ds_root_dir, dataset_cfg['original_dataset_val_dir_name']), 
                        os.path.join(org_ds_root_dir, dataset_cfg['original_dataset_test_dir_name'])]
            dest_dirs = [prep_ds_train_subdir, prep_ds_val_subdir, prep_ds_test_subdir]

            for ds_part, img_list, src_dir, dest_dir in zip(train_val_test, img_lists, src_dirs, dest_dirs):
                if (ds_part == 'validation') and (val_from_train == True):
                    continue
                exist_prep_img_num, completed_new_prep_img_num, failed_new_prep_img_num = eyeq_process(dataset_part=ds_part, image_list=img_list, source_dir=src_dir, destination_dir=dest_dir)
                logging.info('{}, {}, {} images of existed, completed, failed preprocessing for \'{}\'.'.format(exist_prep_img_num, completed_new_prep_img_num, failed_new_prep_img_num, ds_part))

        else: # case of the preprocessed images of the dataset existed
            # Count a number of the train/validation/test of the preprocessed images of the dataset
            train_img_num = len(fnmatch.filter(os.listdir(prep_ds_train_subdir), '*.jpeg'))
            val_img_num = 0
            if val_from_train == False:
                val_img_num = len(fnmatch.filter(os.listdir(prep_ds_val_subdir), '*.jpeg'))
            test_img_num = len(fnmatch.filter(os.listdir(prep_ds_test_subdir), '*.jpeg'))
            logging.info('The preprocessed images of dataset already existed. Found {}, {}, {} images for train and test'.format(train_img_num, val_img_num, test_img_num)) 
    else:
        raise ValueError('Unknown \'{}\' dataset'.format(dataset_name))

    return None, None, None