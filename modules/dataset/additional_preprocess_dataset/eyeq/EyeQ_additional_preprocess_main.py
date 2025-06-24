from absl import logging


def eyeq_additional_preprocess(input_root_dir, list_input_subdir, output_root_dir, addPrep_method, addPrep_method_configs):
    """Additional preprocessing image after doing the preprocessed images for EyeQ from the original images of EyePACS dataset"""
    ### TEST ###
    print('#'*100)
    print('input_root_dir: {}'.format(input_root_dir))
    print('list_input_subdir: {}'.format(list_input_subdir))
    print('output_root_dir: {}'.format(output_root_dir))
    print('addPrep_method: {}'.format(addPrep_method))
    print('addPrep_method_configs: {}'.format(addPrep_method_configs))
    print('#'*100)
    ### TEST ###