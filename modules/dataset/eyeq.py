import os
import random
import numpy as np
import tensorflow as tf


def _EyeQimgPathLabelDict(ds_part_dir, img_info, img_filename_col_name, img_label_col_name, img_path_dict_key, img_label_dict_key):
    # create dictionary of image path and label
    img_path_label_dict = {img_path_dict_key: list(), img_label_dict_key: list()}

    for img in img_info:
        img_path = os.path.join(ds_part_dir, img[img_filename_col_name])
        if os.path.exists(img_path):
            img_path_label_dict[img_path_dict_key].append(img_path)
            img_path_label_dict[img_label_dict_key].append(int(img[img_label_col_name]))

    return img_path_label_dict


def gen_EyeQImgPathLabelDict(ds_num_class, ds_parts, ds_part_dirs, img_infos, img_filename_col_name, img_label_col_name, img_path_dict_key, img_label_dict_key, val_from_train, val_from_train_shuffle, val_portion):
    """Create dictionaries of image path and label for train/validation/test of dataset"""
    # train and validation dataset
    train_img_path_label_dict = _EyeQimgPathLabelDict(ds_part_dir=ds_part_dirs[ds_parts['train']], 
                                                      img_info=img_infos[ds_parts['train']], 
                                                      img_filename_col_name=img_filename_col_name, 
                                                      img_label_col_name=img_label_col_name,
                                                      img_path_dict_key=img_path_dict_key,
                                                      img_label_dict_key=img_label_dict_key)

    # whether validation dataset splitted from train dataset
    if val_from_train == True:
        tmp_val_img_path_label_dict = {img_path_dict_key: list(), img_label_dict_key: list()}
        tmp_train_img_path_label_dict = {img_path_dict_key: list(), img_label_dict_key: list()}

        for class_id in range(ds_num_class):
            indices_of_this_class_train_img_path_label_dict = np.where(np.array(train_img_path_label_dict[img_label_dict_key]) == class_id)[0]
                        
            if val_from_train_shuffle:
                np.random.shuffle(indices_of_this_class_train_img_path_label_dict)
            
            num_train_img_of_this_class = len(indices_of_this_class_train_img_path_label_dict)
            indices_of_this_class_for_val, indices_of_this_class_for_train = np.split(indices_of_this_class_train_img_path_label_dict, 
                                                                                      [int(round(num_train_img_of_this_class*val_portion))])
            
            tmp_val_img_path_label_dict[img_path_dict_key].extend(np.take(np.array(train_img_path_label_dict[img_path_dict_key]), indices_of_this_class_for_val))
            tmp_val_img_path_label_dict[img_label_dict_key].extend(np.take(np.array(train_img_path_label_dict[img_label_dict_key]), indices_of_this_class_for_val))

            tmp_train_img_path_label_dict[img_path_dict_key].extend(np.take(np.array(train_img_path_label_dict[img_path_dict_key]), indices_of_this_class_for_train))
            tmp_train_img_path_label_dict[img_label_dict_key].extend(np.take(np.array(train_img_path_label_dict[img_label_dict_key]), indices_of_this_class_for_train))

        val_img_path_label_dict = tmp_val_img_path_label_dict 
        train_img_path_label_dict = tmp_train_img_path_label_dict    
    else:
        val_img_path_label_dict = _EyeQimgPathLabelDict(ds_part_dir=ds_part_dirs[ds_parts['val']], 
                                                        img_info=img_infos[ds_parts['val']], 
                                                        img_filename_col_name=img_filename_col_name, 
                                                        img_label_col_name=img_label_col_name,
                                                        img_path_dict_key=img_path_dict_key,
                                                        img_label_dict_key=img_label_dict_key)   

    # test dataset
    test_img_path_label_dict = _EyeQimgPathLabelDict(ds_part_dir=ds_part_dirs[ds_parts['test']], 
                                                     img_info=img_infos[ds_parts['test']], 
                                                     img_filename_col_name=img_filename_col_name, 
                                                     img_label_col_name=img_label_col_name,
                                                     img_path_dict_key=img_path_dict_key,
                                                     img_label_dict_key=img_label_dict_key) 

    # shuffle train dataset
    zipped_train_img_path_label_dict = list(zip(train_img_path_label_dict[img_path_dict_key], train_img_path_label_dict[img_label_dict_key]))
    random.shuffle(zipped_train_img_path_label_dict)
    train_img_path_label_dict[img_path_dict_key], train_img_path_label_dict[img_label_dict_key] = zip(*zipped_train_img_path_label_dict)

    # dictionary of image path and label for train/validation/test of dataset
    img_path_label_dicts = {ds_parts['train']: train_img_path_label_dict, 
                            ds_parts['val']: val_img_path_label_dict, 
                            ds_parts['test']: test_img_path_label_dict}

    return img_path_label_dicts


def gen_EyeQdataset(train_val_test, img_path_list):
    """Create dataloader for train/validation/test of dataset""" 
    num_img = len(img_path_list)
    dataset = tf.data.Dataset.from_tensor_slices(img_path_list)

    return dataset, num_img