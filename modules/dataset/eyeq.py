import os
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers


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
    shuffled_train_img_path, shuffled_train_img_label = zip(*zipped_train_img_path_label_dict)
    train_img_path_label_dict[img_path_dict_key] = list(shuffled_train_img_path)
    train_img_path_label_dict[img_label_dict_key] = list(shuffled_train_img_label)

    # dictionary of image path and label for train/validation/test of dataset
    img_path_label_dicts = {ds_parts['train']: train_img_path_label_dict, 
                            ds_parts['val']: val_img_path_label_dict, 
                            ds_parts['test']: test_img_path_label_dict}

    return img_path_label_dicts


def _transform_data_EyeQdataset(train_val_test, augment, augment_infos, resized_img_dim):
    def transform_data(img_path, img_label):
        img = tf.io.read_file(img_path)
        # decode image to tensor with type uint8 having values in the range [0, 255]
        img = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.uint8, expand_animations=False)
        # convert image from type uint8 to float32 which the values are normalized by 255 (MAX value of the input dtype, uint8),
        # and return values in the range [0,1]
        img = tf.image.convert_image_dtype(img, tf.dtypes.float32)
        img = tf.image.resize(img, [resized_img_dim, resized_img_dim])
        if augment:
            data_augmentation = tf.keras.Sequential()
            if augment_infos['hflip']:
                data_augmentation.add(layers.RandomFlip('horizontal'))
            if augment_infos['vflip']:
                data_augmentation.add(layers.RandomFlip('vertical'))
            if augment_infos['rotation']:
                data_augmentation.add(layers.RandomRotation(augment_infos['rotation_factor']))
            img = data_augmentation(img)
        # rescale an input in the [0, 1] range to be in the [-1, 1] range (outputs: `(inputs * scale) + offset`)
        img = layers.Rescaling(scale=2, offset=-1)(img)
        
        return img, img_path, img_label
    return transform_data


def gen_EyeQdataset(train_val_test, img_path_label_dict, img_path_dict_key, img_label_dict_key, shuffle, augment, augment_infos, resized_img_dim, batch_size, drop_remainder=True):
    """Create dataloader for train/validation/test of dataset"""
    num_img_path = len(img_path_label_dict[img_path_dict_key])
    num_img_label = len(img_path_label_dict[img_label_dict_key])
    if num_img_path != num_img_label:
        raise ValueError('Number of image path ({}) and label ({}) of \'{}\' dataset are not equal'.format(num_img_path, num_img_label, train_val_test))

    num_img = num_img_label
    dataset = tf.data.Dataset.from_tensor_slices((img_path_label_dict[img_path_dict_key], img_path_label_dict[img_label_dict_key]))

    if train_val_test == 'train':
        dataset = dataset.repeat()

    if shuffle:
        dataset = dataset.shuffle(buffer_size=num_img)

    dataset = dataset.map(
                _transform_data_EyeQdataset(train_val_test, augment, augment_infos, resized_img_dim),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset, num_img
