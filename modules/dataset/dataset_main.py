from absl import logging
import os
import fnmatch
import tensorflow as tf

from modules.utils import (read_csv_to_list_of_dict)
from modules.dataset.preprocess_dataset.eyeq.EyeQ_process_main import (eyeq_process)
from modules.dataset.eyeq import (gen_EyeQImgPathLabelDict, gen_EyeQdataset)

def gen_dataset(dataset_name, dataset_cfg):
    """Generate train/validation/test dataloader of the dataset"""

    if dataset_name == 'EyeQ': # Eye Quality (EyeQ) dataset
        # get information of dataset
        num_class = dataset_cfg['class_num']
        name_class = dataset_cfg['class_name']

        # get details of data preprocessing
        val_from_train = dataset_cfg['val_from_train']
        val_from_train_shuffle = dataset_cfg['val_from_train_shuffle']
        val_portion = dataset_cfg['val_portion']

        # check if the preprocessed images of the dataset existed
        # if not existing, the preprocessed images firstly need to be created
        ds_root_dir = dataset_cfg['dataset_root_dir']
        prep_ds_root_dir = os.path.join(ds_root_dir, dataset_cfg['dataset_preprocessed_dir_name'])
        prep_ds_train_subdir = os.path.join(prep_ds_root_dir, dataset_cfg['dataset_preprocessed_train_dir_name'])
        prep_ds_val_subdir = os.path.join(prep_ds_root_dir, dataset_cfg['dataset_preprocessed_val_dir_name'])
        prep_ds_test_subdir = os.path.join(prep_ds_root_dir, dataset_cfg['dataset_preprocessed_test_dir_name'])

        # Read list of train/val/test images for EyeQ which is a subset of all train/val/test images of EyePACS dataset
        # these lists get from the original source code of EyeQ dataset project
        # the csv file for train/val/test image list of EyeQ dataset has [id, image, quality, DR_grade] columns 
        # the format of train/val/test images info is a list of dictionaries
        ds_src_dir = os.path.join(ds_root_dir, dataset_cfg['dataset_src_dir_name'])
        ds_src_train_val_test_list_dir = os.path.join(ds_src_dir, dataset_cfg['dataset_src_train_val_test_list_dir_name'])
        train_img_info_csv = os.path.join(ds_src_train_val_test_list_dir, dataset_cfg['dataset_src_train_list_filename'])
        val_img_info_csv = os.path.join(ds_src_train_val_test_list_dir, dataset_cfg['dataset_src_val_list_filename'])
        test_img_info_csv = os.path.join(ds_src_train_val_test_list_dir, dataset_cfg['dataset_src_test_list_filename'])

        train_img_info = read_csv_to_list_of_dict(train_img_info_csv, ',')
        val_img_info = list()
        if val_from_train == False:
            val_img_info = read_csv_to_list_of_dict(val_img_info_csv, ',')    
        test_img_info = read_csv_to_list_of_dict(test_img_info_csv, ',')
        logging.info('Found the number of images for train, validation, test: {}, {}, {}'.format(len(train_img_info), len(val_img_info), len(test_img_info)))

        if not os.path.exists(prep_ds_root_dir): # case of the preprocessed images of the dataset not existed
            logging.info('The preprocessed images of dataset cannot be found. Starting to create....') 
            
            # Preprocess images of EyeQ dataset from EyePACS dataset
            org_ds_root_dir = dataset_cfg['original_dataset_root_dir']
            train_val_test = ['train', 'validation', 'test']
            img_infos = [train_img_info, val_img_info, test_img_info]
            src_dirs = [os.path.join(org_ds_root_dir, dataset_cfg['original_dataset_train_dir_name']), 
                        os.path.join(org_ds_root_dir, dataset_cfg['original_dataset_val_dir_name']), 
                        os.path.join(org_ds_root_dir, dataset_cfg['original_dataset_test_dir_name'])]
            dest_dirs = [prep_ds_train_subdir, prep_ds_val_subdir, prep_ds_test_subdir]

            for ds_part, img_info, src_dir, dest_dir in zip(train_val_test, img_infos, src_dirs, dest_dirs):
                if (ds_part == 'validation') and (val_from_train == True):
                    continue
                exist_prep_img_num, completed_new_prep_img_num, failed_new_prep_img_num = eyeq_process(dataset_part=ds_part, image_info=img_info, source_dir=src_dir, destination_dir=dest_dir)
                logging.info('{}, {}, {} images of existed, completed, failed preprocessing for \'{}\'.'.format(exist_prep_img_num, completed_new_prep_img_num, failed_new_prep_img_num, ds_part))

        else: # case of the preprocessed images of the dataset existed
            # Count a number of the train/validation/test of the preprocessed images of the dataset
            train_img_num = len(fnmatch.filter(os.listdir(prep_ds_train_subdir), '*.jpeg'))
            val_img_num = 0
            if val_from_train == False:
                val_img_num = len(fnmatch.filter(os.listdir(prep_ds_val_subdir), '*.jpeg'))
            test_img_num = len(fnmatch.filter(os.listdir(prep_ds_test_subdir), '*.jpeg'))
            logging.info('The preprocessed images of dataset already existed. Found {}, {}, {} images for train, validation, and test'.format(train_img_num, val_img_num, test_img_num))

        # At this point, the preprocessed images of the dataset already used, then
        # [1st step] create dictionaries of image path and label for train/validation/test of dataset, then
        # [2nd step] generate train/validation/test dataloader of the EyeQ dataset

        # !------ [1st step] ------!
        ds_parts = {'train': 'train', 
                    'val': 'validation', 
                    'test': 'test'}

        prep_ds_dirs = {ds_parts['train']: prep_ds_train_subdir, 
                        ds_parts['val']: prep_ds_val_subdir, 
                        ds_parts['test']: prep_ds_test_subdir}

        img_infos = {ds_parts['train']: train_img_info, 
                     ds_parts['val']: val_img_info, 
                     ds_parts['test']: test_img_info}

        img_filename_col_name = dataset_cfg['dataset_csv_info_column_name_of_image_filename']
        img_label_col_name = dataset_cfg['dataset_csv_info_column_name_of_image_label']
        
        # dictionraies of image path and label for train/validation/test of dataset
        img_path_label_dicts = gen_EyeQImgPathLabelDict(ds_num_class=num_class,
                                                        ds_parts=ds_parts, 
                                                        ds_part_dirs=prep_ds_dirs,
                                                        img_infos=img_infos,
                                                        img_filename_col_name=img_filename_col_name,
                                                        img_label_col_name=img_label_col_name,
                                                        val_from_train=val_from_train,
                                                        val_from_train_shuffle=val_from_train_shuffle,
                                                        val_portion=val_portion)

        # # train/validation/test dataloader of the EyeQ dataset
        # ds_dataloaders = list()
        # for (k_ds_part, v_ds_part), (k_img_path_list, v_img_path_list) in zip(ds_parts.items(), img_path_lists.items()):
        #      ds_dataloader, total_num_img = gen_EyeQdataset(train_val_test=v_ds_part, img_path_list=v_img_path_list)
        #      ds_dataloaders.append(ds_dataloader)
        #      logging.info('Create \'{}\' dataloader containing {} images'.format(v_ds_part, total_num_img))

    else:
        raise ValueError('Unknown \'{}\' dataset'.format(dataset_name))

    return None, None, None