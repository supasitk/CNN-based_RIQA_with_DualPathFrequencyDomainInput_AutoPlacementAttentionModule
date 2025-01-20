import tensorflow as tf
from tensorflow.keras.layers import (Rescaling, Lambda)

import modules.dataset.utils.cv_functional as cv_functional
from modules.model.layer.CustomLayer import (ImageRGBtoYUV, Depthwise_DCT2d)

__all__ = ["Compose", "Resize", "RescaleValue", "ColorSpaceConversion", "DepthwiseDCT2d", "ExpandDims", "SqueezeDims"]

class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.Resize(...),
        >>>     transforms.Rescale(...),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Resize(object):
    """Resize the input Image to the given size.

    Args:
        size (int): Desired output size. 
        interpolation (int, optional): Desired interpolation. Default is ``BILINEAR``
    """

    def __init__(self, size, interpolation='BILINEAR'):
        assert isinstance(size, int)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (np.ndarray): Image to be scaled.

        Returns:
            np.ndarray: Rescaled image.
        """
        return cv_functional.resize(img, self.size, self.interpolation)
    

class RescaleValue(object):
    """Rescale value of the input Image to specific range"""

    def __init__(self, rescale_range):
        if rescale_range == 'ZeroToOne':
            scale_value = 1./255
            offset_value = 0
        elif rescale_range == 'minusOneToOne':
            scale_value = 1./127.5
            offset_value = -1
        else:
            raise ValueError
        
        self.rescale_value = Rescaling(scale=scale_value, offset=offset_value)
    
    def __call__(self, img):
        return self.rescale_value(img)
    

class ColorSpaceConversion(object):
    """Color space conversion"""

    def __init__(self, color_space_conversion):
        if color_space_conversion == None:
            self.color_space_conversion_fn = Lambda(lambda x: x)
        elif color_space_conversion == 'YUV':
            self.color_space_conversion_fn = ImageRGBtoYUV()
        else:
            raise ValueError
        
    def __call__(self, img):
        return self.color_space_conversion_fn(img)
    

class DepthwiseDCT2d(object):
    """Depthwise DCT2d"""

    def __init__(self, dct_infos):
        self.dct = Depthwise_DCT2d(tile_size_x=dct_infos['tile_size_x'], 
                                   tile_size_y=dct_infos['tile_size_y'],
                                   freq_component_set=dct_infos['freq_component_set'],
                                   freq_component_selection_of_each_channel=dct_infos['freq_component_selection_of_each_channel'],
                                   num_selected_freq_component_of_each_channel=dct_infos['num_selected_freq_component_of_each_channel'])
        
    def __call__(self, img):
        return self.dct(img)
    

class ExpandDims(object):
    """Returns a tensor with a length 1 axis inserted at index axis"""

    def __init__(self, axis):
        self.axis = axis

    def __call__(self, img):
        return tf.expand_dims(img, axis=self.axis)


class SqueezeDims(object):
    """Removes dimensions of size 1 from the shape of a tensor, you can remove specific size 1 dimensions by specifying axis"""

    def __init__(self, axis):
        self.axis = axis

    def __call__(self, img):
        return tf.squeeze(img, axis=self.axis)