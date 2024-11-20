import os
import random
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

from tensorflow.keras.layers import (RandomFlip, RandomRotation, Rescaling)


def _EyeQimgPathLabelDict(ds_part_dir, img_info, img_filename_col_name, img_label_col_name, img_path_dict_key, img_label_dict_key):
    # create dictionary of image path and label
    img_path_label_dict = {img_path_dict_key: list(), img_label_dict_key: list()}

    for img in img_info:
        img_path = os.path.join(ds_part_dir, img[img_filename_col_name])
        if os.path.exists(img_path):
            img_path_label_dict[img_path_dict_key].append(img_path)
            img_path_label_dict[img_label_dict_key].append(int(img[img_label_col_name]))

    return img_path_label_dict


def gen_EyeQImgPathLabelDict(ds_num_class, ds_parts, ds_part_dirs, img_infos, img_filename_col_name, img_label_col_name, img_path_dict_key, img_label_dict_key, 
                             train_portion, train_shuffle, val_from_train, val_from_train_shuffle, val_portion, seed=None):
    """Create dictionaries of image path and label for train/validation/test of dataset"""
    # train and validation dataset
    train_img_path_label_dict = _EyeQimgPathLabelDict(ds_part_dir=ds_part_dirs[ds_parts['train']], 
                                                      img_info=img_infos[ds_parts['train']], 
                                                      img_filename_col_name=img_filename_col_name, 
                                                      img_label_col_name=img_label_col_name,
                                                      img_path_dict_key=img_path_dict_key,
                                                      img_label_dict_key=img_label_dict_key)

    if train_portion != 1.0: # all training data are not used
        tmp_train_img_path_label_dict = {img_path_dict_key: list(), img_label_dict_key: list()}
        
        for class_id in range(ds_num_class):
            indices_of_this_class_train_img_path_label_dict = np.where(np.array(train_img_path_label_dict[img_label_dict_key]) == class_id)[0]
        
            if train_shuffle:
                indices_of_this_class_train_img_path_label_dict = tf.random.shuffle(indices_of_this_class_train_img_path_label_dict, seed=seed)
        
            num_train_img_of_this_class = len(indices_of_this_class_train_img_path_label_dict)
            indices_of_this_class_for_train, _ = np.split(indices_of_this_class_train_img_path_label_dict, 
                                                          [int(round(num_train_img_of_this_class*train_portion))])
                                                    
            tmp_train_img_path_label_dict[img_path_dict_key].extend(np.take(np.array(train_img_path_label_dict[img_path_dict_key]), indices_of_this_class_for_train))
            tmp_train_img_path_label_dict[img_label_dict_key].extend(np.take(np.array(train_img_path_label_dict[img_label_dict_key]), indices_of_this_class_for_train))

        train_img_path_label_dict = tmp_train_img_path_label_dict            

    # whether validation dataset splitted from train dataset
    if val_from_train == True:
        tmp_val_img_path_label_dict = {img_path_dict_key: list(), img_label_dict_key: list()}
        tmp_train_img_path_label_dict = {img_path_dict_key: list(), img_label_dict_key: list()}

        for class_id in range(ds_num_class):
            indices_of_this_class_train_img_path_label_dict = np.where(np.array(train_img_path_label_dict[img_label_dict_key]) == class_id)[0]
                        
            if val_from_train_shuffle:
                indices_of_this_class_train_img_path_label_dict = tf.random.shuffle(indices_of_this_class_train_img_path_label_dict, seed=seed)
            
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

    # train dataset
    zipped_train_img_path_label_dict = list(zip(train_img_path_label_dict[img_path_dict_key], train_img_path_label_dict[img_label_dict_key]))
    # if train_shuffle:
    #     random.shuffle(zipped_train_img_path_label_dict)
    train_img_path, train_img_label = zip(*zipped_train_img_path_label_dict)
    train_img_path_label_dict[img_path_dict_key] = list(train_img_path)
    train_img_path_label_dict[img_label_dict_key] = list(train_img_label)

    # dictionary of image path and label for train/validation/test of dataset
    img_path_label_dicts = {ds_parts['train']: train_img_path_label_dict, 
                            ds_parts['val']: val_img_path_label_dict, 
                            ds_parts['test']: test_img_path_label_dict}

    return img_path_label_dicts


def _transform_data_EyeQdataset(ds_input_types, train_val_test, augment, augment_infos, resized_img_dim, seed=None):
    # function of data augmentation
    if augment:
        data_augmentation = tf.keras.Sequential()
        if augment_infos['hflip']:
            data_augmentation.add(RandomFlip('horizontal', seed=seed))
        if augment_infos['vflip']:
            data_augmentation.add(RandomFlip('vertical', seed=seed))
        if augment_infos['rotation']:
            data_augmentation.add(RandomRotation(augment_infos['rotation_factor'], seed=seed))    

    def transform_data(img_path, img_label):
        list_input_types = list()
        for input_type in ds_input_types:
            if input_type == 'RGB':
                this_input_type = tf.io.read_file(img_path)
                # decode image (default type is 'RGB') to tensor with type uint8 having values in the range [0, 255]
                this_input_type = tf.io.decode_image(this_input_type, channels=3, dtype=tf.dtypes.uint8, expand_animations=False)
            else:
                raise ValueError('Input type of data \'{}\' is unknown'.format(input_type))
            # get and concatenate types of input needed for model 
            list_input_types.append(this_input_type)
        inputs = tf.concat(list_input_types, axis=-1)
        # resize
        inputs = tf.image.resize(inputs, [resized_img_dim, resized_img_dim])
        # augmentation
        if augment:
            inputs = data_augmentation(inputs)
        # rescale an input in the [0, 255] range to be in the [-1, 1] range (outputs: `(inputs * scale) + offset`)
        inputs = Rescaling(scale=1./127.5, offset=-1)(inputs)
        
        return inputs, img_path, img_label
    return transform_data


def gen_EyeQdataset(ds_num_class, ds_input_types, train_val_test, img_path_label_dict, img_path_dict_key, img_label_dict_key, shuffle, augment, augment_infos, 
                    resized_img_dim, batch_size, drop_remainder=True, oversampling=False, oversampling_infos=None, train_search=False, seed=None):
    """Create dataloader for train/validation/test of dataset"""
    num_img_path = len(img_path_label_dict[img_path_dict_key])
    num_img_label = len(img_path_label_dict[img_label_dict_key])
    if num_img_path != num_img_label:
        raise ValueError('Number of image path ({}) and label ({}) of \'{}\' dataset are not equal'.format(num_img_path, num_img_label, train_val_test))
    num_img = num_img_label

    if oversampling:
        if oversampling_infos['type'] == 'WeightedProbabilityEveryClassOnEachBatch':
            # split whole (all class) data into each specfic-class (each class) data, then create list of dataloader for each specific-class data
            list_dataloader_for_each_specific_class_data = list()
            for class_id in range(ds_num_class):
                # -- split whole (all class) data into this specfic-class data
                indices_of_this_class_in_ds_img_path_label_dict = np.where(np.array(img_path_label_dict[img_label_dict_key]) == class_id)[0]
                num_img_of_this_class = len(indices_of_this_class_in_ds_img_path_label_dict)

                this_class_img_path_label_dict = {img_path_dict_key: list(), img_label_dict_key: list()}
                this_class_img_path_label_dict[img_path_dict_key].extend(np.take(np.array(img_path_label_dict[img_path_dict_key]), indices_of_this_class_in_ds_img_path_label_dict))
                this_class_img_path_label_dict[img_label_dict_key].extend(np.take(np.array(img_path_label_dict[img_label_dict_key]), indices_of_this_class_in_ds_img_path_label_dict))

                # -- create dataloader for this specific-class data
                this_class_dataset = tf.data.Dataset.from_tensor_slices((this_class_img_path_label_dict[img_path_dict_key], this_class_img_path_label_dict[img_label_dict_key]))
                if shuffle:
                    this_class_dataset = this_class_dataset.shuffle(buffer_size=num_img_of_this_class, seed=seed)
                this_class_dataset = this_class_dataset.repeat()

                # -- add this specific-class dataloader to list of dataloader for each specific-class data
                list_dataloader_for_each_specific_class_data.append(this_class_dataset)
            
            # create the final dataloader contains every specific-class dataloaders with given probability (weight) of sampling for each specific-class data
            dataset = tf.data.Dataset.sample_from_datasets(datasets=list_dataloader_for_each_specific_class_data, 
                                                           weights=oversampling_infos['weight'],
                                                           seed=seed)
        else:
            raise ValueError('Unknown \'{}\' type of random oversampling'.format(oversampling_infos['type']))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((img_path_label_dict[img_path_dict_key], img_path_label_dict[img_label_dict_key]))

    if ((train_val_test == 'train') or ((train_val_test == 'validation') and (train_search == True))):
        dataset = dataset.repeat()

    if shuffle:
        dataset = dataset.shuffle(buffer_size=num_img, seed=seed)

    dataset = dataset.map(
                _transform_data_EyeQdataset(ds_input_types, train_val_test, augment, augment_infos, resized_img_dim, seed=seed),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset, num_img
