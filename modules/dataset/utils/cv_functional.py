import tensorflow as tf

INTERPOLATION_MODE = {'BILINEAR': tf.image.ResizeMethod.BILINEAR}


def resize(img, size, interpolation='BILINEAR'):
    """Resize the input Image to the given size.

    Args:
        img (np.ndarray): Image to be resized.
        size (tuple or int): Desired output size. 
        interpolation (str, optional): Desired interpolation. Default is ``BILINEAR``

    Returns:
        cv Image: Resized image.
    """
    h = size 
    w = size
    return tf.image.resize(img, size=[int(h), int(w)], method=INTERPOLATION_MODE[interpolation])