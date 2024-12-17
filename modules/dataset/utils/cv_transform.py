from tensorflow.keras.layers import (Rescaling)

import modules.dataset.utils.cv_functional as cv_functional


__all__ = ["Compose", "Resize", "RescaleValue"]

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