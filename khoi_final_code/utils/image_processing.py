from skimage.measure import label

def mask_to_bitmask(mask, background):
    """
    Convert background values to 0 and others to 1
    Parameters:
    --------------------
    mask: numpy arrays of a mask
    background: a value
    Returns:
    --------------------
    A numpy array with the same shape as mask but with only 2 values 0 and 1
    """
    return label(mask != background)
