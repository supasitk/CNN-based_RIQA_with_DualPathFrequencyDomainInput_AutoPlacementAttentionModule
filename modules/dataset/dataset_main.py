from absl import logging
import os
import fnmatch
import tensorflow as tf

from modules.utils import (read_csv_to_list_of_dict)
from modules.dataset.preprocess_dataset.eyeq.EyeQ_process_main import (eyeq_process)
from modules.dataset.additional_preprocess_dataset.eyeq.EyeQ_additional_preprocess_main import (eyeq_additional_preprocess)
from modules.dataset.eyeq import (gen_EyeQImgPathLabelDict, gen_EyeQdataset)

def gen_dataset(dataset_name, dataset_cfg, model_cfg, train_cfg, train_search=False, seed=None):
    """Generate train/validation/test dataloader of the dataset"""

    if dataset_name == 'EyeQ': # Eye Quality (EyeQ) dataset
        # get information of dataset
        num_class = dataset_cfg['class_num']
        name_class = dataset_cfg['class_name']

        # get details of data preprocessing
        train_portion = dataset_cfg['train_portion']
        train_shuffle = dataset_cfg['train_shuffle']
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
        # Additional preprocessing [if any] for the preprocessed images of the dataset
        # check if the additional preprocessing for the preprocessed images of the dataset existed
        # if not existing, the additional preprocessing for the preprocessed images of the dataset firstly need to be done 
        if dataset_cfg['additional_preprocessing_image']:
            # root path of the input directory (including name of 'train/val/test' directory under this) containing the preprocessed images of the dataset for doing the additional preprocessing
            addPrepImg_input_root_dir = dataset_cfg['addPrepImg_input_root_dir']
            addPrepImg_input_train_dir_name = dataset_cfg['addPrepImg_input_train_dir_name']
            addPrepImg_input_val_dir_name = dataset_cfg['addPrepImg_input_val_dir_name']
            addPrepImg_input_test_dir_name = dataset_cfg['addPrepImg_input_test_dir_name']
            # list of name of the 'train/val/test' directory of the root of input directory 
            # removing duplicate dir name if same name found, e.g., if name of 'val' dir is same to name of 'train' dir (val_from_train=True),
            # to avoid repeat of the additional preprocessing on the same directory
            list_addPrepImg_input_train_val_test_dir_name = list(dict.fromkeys([addPrepImg_input_train_dir_name, addPrepImg_input_val_dir_name, addPrepImg_input_test_dir_name]))

            # root path of the output directory for doing the additional preprocessing
            addPrepImg_output_root_dir = dataset_cfg['addPrepImg_output_root_dir']

            # list of method(s) for the additional preprocessing
            addPrepImg_methods = dataset_cfg['addPrepImg_methods']
            logging.info('Additional preprocessing \'{}\' for the preprocessed images of dataset required.'.format(addPrepImg_methods))

            # doing each method(s) of additional preprocessing
            for additional_prep_method in addPrepImg_methods:
                if additional_prep_method == 'RGB_wCLAHE':
                    # root path of output directory ('addPrepImg_output_root_dir' joined with 'addPrepImg_{method}_dir_name' ) for keeping additional preprocessing image of this method
                    addPrepImg_method_output_root_dir = os.path.join(addPrepImg_output_root_dir, dataset_cfg['addPrepImg_RGBwCLAHE_dir_name'])

                    # configuration of this method for additional preprocessing 
                    addPrepImg_method_configs = dataset_cfg['addPrepImg_RGBwCLAHE_config']
                else:
                    raise ValueError('Unknown \'{}\' additional preprocessing method for the preprocessed images of dataset'.format(additional_prep_method))
                
                if not os.path.exists(addPrepImg_method_output_root_dir): # case of the additional preprocessing for the preprocessed images of the dataset not existed
                    logging.info('Additional preprocessing \'{}\' for the preprocessed images of dataset cannot be found. Starting to create....'.format(additional_prep_method))
                    # doing this additional processing specified
                    eyeq_additional_preprocess(input_root_dir=addPrepImg_input_root_dir, 
                                               list_input_subdir=list_addPrepImg_input_train_val_test_dir_name, 
                                               output_root_dir=addPrepImg_method_output_root_dir, 
                                               addPrep_method=additional_prep_method, 
                                               addPrep_method_configs=addPrepImg_method_configs)
                else: # case of the additional preprocessing for the preprocessed images of the dataset existed
                    # count a number of the train/validation/test images of this additional preprocessing for the preprocessed images of the dataset
                    addPrepImg_train_img_num = len(fnmatch.filter(os.listdir(os.path.join(addPrepImg_method_output_root_dir, addPrepImg_input_train_dir_name)), '*.jpeg'))
                    addPrepImg_val_img_num = 0
                    if addPrepImg_input_val_dir_name != addPrepImg_input_train_dir_name:
                        addPrepImg_val_img_num = len(fnmatch.filter(os.listdir(os.path.join(addPrepImg_method_output_root_dir, addPrepImg_input_val_dir_name)), '*.jpeg'))
                    addPrepImg_test_img_num = len(fnmatch.filter(os.listdir(os.path.join(addPrepImg_method_output_root_dir, addPrepImg_input_test_dir_name)), '*.jpeg'))
                    logging.info('Additional preprocessing \'{}\' for the preprocessed images of dataset already existed. Found {}, {}, {} images for train, validation, and test'.format(additional_prep_method, addPrepImg_train_img_num, addPrepImg_val_img_num, addPrepImg_test_img_num))

        # At this point, the preprocessed (including additional_preprocessed [if any]) images of the dataset already used, then
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

        img_path_dict_key = 'img_path'
        img_label_dict_key = 'img_label'
        
        # dictionaries of image path and label for train/validation/test of dataset
        img_path_label_dicts = gen_EyeQImgPathLabelDict(ds_num_class=num_class,
                                                        ds_parts=ds_parts, 
                                                        ds_part_dirs=prep_ds_dirs,
                                                        img_infos=img_infos,
                                                        img_filename_col_name=img_filename_col_name,
                                                        img_label_col_name=img_label_col_name,
                                                        img_path_dict_key=img_path_dict_key,
                                                        img_label_dict_key=img_label_dict_key,
                                                        train_portion=train_portion, 
                                                        train_shuffle=train_shuffle,
                                                        val_from_train=val_from_train,
                                                        val_from_train_shuffle=val_from_train_shuffle,
                                                        val_portion=val_portion,
                                                        seed=seed)

        # !------ [2nd step] ------!
        ds_dataloaders = list()
        ds_num_imgs = list()
        for (k_ds_part, v_ds_part), (k_img_path_label_dict, v_img_path_label_dict) in zip(ds_parts.items(), img_path_label_dicts.items()):
            if ((v_ds_part == 'train') or ((v_ds_part == 'validation') and (train_search == True))):
                shuffle = True
                augment = dataset_cfg['train_augment']
                augment_infos = None
                if augment:
                    augment_infos = dict()
                    augment_infos.update([('hflip', dataset_cfg['train_augment_hflip'])])
                    augment_infos.update([('vflip', dataset_cfg['train_augment_vflip'])])
                    augment_infos.update([('rotation', dataset_cfg['train_augment_rotation'])])
                    augment_infos.update([('rotation_factor', dataset_cfg['train_augment_rotation_factor'])])
                drop_remainder = True
                oversampling = dataset_cfg['train_random_oversampling']
                oversampling_infos = None
                if oversampling:
                    oversampling_infos = dict()
                    oversampling_infos.update([('type', dataset_cfg['train_random_oversampling_type'])])
                    oversampling_infos.update([('weight', dataset_cfg['train_random_oversampling_weight'])])
            else:
                shuffle = False
                augment = False
                augment_infos = None
                drop_remainder = False
                oversampling = False
                oversampling_infos = None

            rescale = dataset_cfg['rescale_input_value']
            rescale_infos = None
            if rescale:
                rescale_infos = dict()
                rescale_infos.update([('range', dataset_cfg['rescale_input_value_range'])])

            normalize = dataset_cfg['normalization']
            normalize_infos = None
            if normalize:
                normalize_infos = dict()
                normalize_infos.update([('stat', dataset_cfg['normalization_stat'])])

            ds_dataloader, total_num_img = gen_EyeQdataset(ds_num_class=num_class,
                                                           ds_input_types=model_cfg['input_types_for_model'],
                                                           train_val_test=v_ds_part, 
                                                           img_path_label_dict=v_img_path_label_dict,
                                                           img_path_dict_key=img_path_dict_key,
                                                           img_label_dict_key=img_label_dict_key,
                                                           shuffle=shuffle,
                                                           augment=augment,
                                                           augment_infos=augment_infos,
                                                           resized_img_dim=dataset_cfg['resized_image_dim'],
                                                           rescale=rescale,
                                                           rescale_infos=rescale_infos,
                                                           normalize=normalize,
                                                           normalize_infos=normalize_infos,
                                                           batch_size=train_cfg['batch_size'],
                                                           drop_remainder=drop_remainder,
                                                           oversampling=oversampling, 
                                                           oversampling_infos=oversampling_infos,
                                                           train_search=train_search,
                                                           seed=seed)
            ds_dataloaders.append(ds_dataloader)
            this_ds_num_img_dict = {'total_num_img': total_num_img, 'list_num_img_of_each_class_id': list()}
            logging.info('Create \'{}\' dataloader containing {} images (Number and portion of data for each class are below)'.format(v_ds_part, total_num_img))
            # show number and portion of data for each class of dataset
            assert num_class == len(name_class)
            str_num_and_portion_of_data_for_classes_of_ds = ""
            for class_id, class_name in enumerate(name_class):
                num_input_of_this_class = v_img_path_label_dict[img_label_dict_key].count(class_id)
                this_ds_num_img_dict['list_num_img_of_each_class_id'].append(num_input_of_this_class)
                portion_input_of_this_class = num_input_of_this_class / total_num_img * 100
                str_num_and_portion_of_data_for_classes_of_ds += f"{name_class[class_id]} [{num_input_of_this_class} ({portion_input_of_this_class:.2f}%)], "
            logging.info('\t{}'.format(str_num_and_portion_of_data_for_classes_of_ds))
            ds_num_imgs.append(this_ds_num_img_dict)
    else:
        raise ValueError('Unknown \'{}\' dataset'.format(dataset_name))

    return (ds_dataloaders[0], ds_num_imgs[0],       
            ds_dataloaders[1], ds_num_imgs[1],      
            ds_dataloaders[2], ds_num_imgs[2])      # train, validation, test
