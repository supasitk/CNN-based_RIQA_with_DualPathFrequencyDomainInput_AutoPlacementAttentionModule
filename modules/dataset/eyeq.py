import os
import tensorflow as tf


def gen_EyeQdataset(train_val_test, img_path_list):
    """Create dataloader for train/validation/test of dataset""" 
    num_img = len(img_path_list)
    dataset = tf.data.Dataset.from_tensor_slices(img_path_list)

    return dataset, num_img