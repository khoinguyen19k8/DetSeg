"""
Plotting utilities
"""
from matplotlib.patches import Rectangle
from .lesion_tools import get_lesion_props

def bbox_overlay(mask, edgecolor, linewidth):
    """
    Read a mask file and return a list of Rectangles for plotting. 
    Please read documentation for matplotlib.patches.Rectangle for edgecolor and linewidth parameters.
    """
    lesion_props = get_lesion_props(mask, 0.0)
    recs = []
    for lesion_region in lesion_props:
        x_anchor, y_anchor = lesion_region.bbox[1], lesion_region.bbox[0]
        bbox_height, bbox_width = lesion_region.image.shape
        recs.append(Rectangle((x_anchor, y_anchor), bbox_width, bbox_height, edgecolor = edgecolor, linewidth = linewidth))
    return recs
