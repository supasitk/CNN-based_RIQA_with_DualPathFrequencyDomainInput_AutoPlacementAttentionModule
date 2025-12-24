import cv2


class _cv2_RGB_wCLAHE:
    def __init__(self, clipLimit, tileGridSize):
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize

        self.clahe = cv2.createCLAHE(clipLimit=self.clipLimit, 
                                     tileGridSize=self.tileGridSize)

    def apply(self, input_img_path, output_img_path):
        # read input image (get BGR image by default)
        bgr_img = cv2.imread(input_img_path) 
        # convert to LAB color space
        lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        # split the LAB image into channels
        l_channel, a_channel, b_channel = cv2.split(lab_img)
        # apply CLAHE to the L channel
        clahe_l_channel = self.clahe.apply(l_channel)
        # merge the enhanced L channel with original A and B channels
        clahe_lab_img = cv2.merge([clahe_l_channel, a_channel, b_channel])
        # convert back to BGR color space
        clahe_bgr_img = cv2.cvtColor(clahe_lab_img, cv2.COLOR_LAB2BGR)
        # write output image
        cv2.imwrite(output_img_path, clahe_bgr_img)


def gen_addPrepImg_funtion(addPrep_method, addPrep_method_configs):
    if addPrep_method == 'RGB_wCLAHE':
        addPrep_method_configs_library = addPrep_method_configs['library']
        if addPrep_method_configs_library == 'cv2':
            clipLimit = addPrep_method_configs['clipLimit']
            tileGridSize = addPrep_method_configs['tileGridSize']

            addPrepImg_function = _cv2_RGB_wCLAHE(clipLimit=clipLimit,
                                                  tileGridSize=(tileGridSize, tileGridSize))
        else:
            raise ValueError('Unknown \'{}\' library used for additional preprocessing method \'{}\' for the preprocessed images of dataset'.format(addPrep_method_configs_library, addPrep_method))
    else:
        raise ValueError('Unknown \'{}\' additional preprocessing method for the preprocessed images of dataset'.format(addPrep_method))
    
    return addPrepImg_function