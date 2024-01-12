import os
import random
import numpy as np


def _imgPathList(ds_part_dir, file_type=".jpeg"):
    # create list of image path
    img_path_list = list()

    for img_name in os.listdir(ds_part_dir):
        if img_name.endswith(file_type):
            img_path_list.append(os.path.join(ds_part_dir, img_name))

    return img_path_list


def gen_ImgPathList(ds_parts, ds_part_dirs, val_from_train, val_from_train_shuffle, val_portion, file_type=".jpeg"):
    """Create list of image path for train/validation/test of dataset"""
    # train and validation dataset
    train_img_path_list = _imgPathList(ds_part_dir=ds_part_dirs[ds_parts['train']])
    # whether validation dataset splitted from train dataset
    if val_from_train == True:
        if val_from_train_shuffle:
            random.shuffle(train_img_path_list)
        
        num_train_img = len(train_img_path_list)
        val_img_path_list, train_img_path_list = np.split(train_img_path_list, [int(round(num_train_img*val_portion))])
    else:
        val_img_path_list = _imgPathList(ds_part_dir=ds_part_dirs[ds_parts['val']])

    # test dataset
    test_img_path_list = _imgPathList(ds_part_dir=ds_part_dirs[ds_parts['test']])

    img_path_lists = {ds_parts['train']: train_img_path_list, 
                      ds_parts['val']: val_img_path_list, 
                      ds_parts['test']: test_img_path_list}

    return img_path_lists