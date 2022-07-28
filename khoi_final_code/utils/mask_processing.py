from skimage.measure import label 

def create_instance_mask(mask, background):
    """
    Given a segmentation mask, create a binary mask for each instance.
    --------------------
    Parameters:
    
    mask: The mask image array.
    background: Background value
    --------------------
    Returns:

    A list consists of instances mask, each as a numpy array. If there is no object in the mask, simply return 0.
    """
    mask_map, count = label(mask != background, return_num=True)
    if count == 0:
        return count
    instances = []
    for instance_num in range(1, count + 1):
        instance_mask = label(mask_map == instance_num)
        instances.append(instance_mask)
    return instances