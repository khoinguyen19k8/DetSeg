import skimage
from skimage.measure import label, regionprops

def get_lesion_props(mask_arr, background):
    """
    mask_arr: A mask array
    background: What value should be considered background.
    """
    mask_lab = label(mask_arr != background)
    lesion_props = regionprops(mask_lab)
    return lesion_props 

def normalize_lesions_props(center, bbox, img_size):
    """
    Parameters
    ----------
    center: (x_center, y_center), a tuple.
    bbox: (bbox_width, bbox_height), a tuple.
    img_size: (img_width, img_height), a tuple. 
    """
    x_center_norm = center[0] / img_size[0] 
    y_center_norm = center[1] / img_size[1] 
    bbox_width_norm = bbox[0] / img_size[0] 
    bbox_height_norm = bbox[1] / img_size[1]

    return [(x_center_norm, y_center_norm), (bbox_width_norm, bbox_height_norm)] 

def unnormalize_lesion_props(center, bbox, img_size):
    """
    Parameters
    ----------
    center: (x_center, y_center) in normalized format, ranged from 0 to 1
    bbox: (bbox_width, bbox_height) in normalized format, ranged from 0 to 1.
    img_size: (img_width, img_height), a tuple. 
    """
    x_center_unnorm = center[0] * img_size[0] 
    y_center_unnorm = center[1] * img_size[1] 
    bbox_width_unnorm = bbox[0] * img_size[0] 
    bbox_height_unnorm = bbox[1] * img_size[1]
    
    return [(x_center_unnorm, y_center_unnorm), (bbox_width_unnorm, bbox_height_unnorm)] 

