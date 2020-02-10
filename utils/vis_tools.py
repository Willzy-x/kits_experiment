import cv2
import os
import scipy
import numpy as np
from pathlib import Path
from starter_code.utils import load_volume, load_segmentation

# colors (BGR)
CYAN     = (255,255,0)
MAGENTA  = (128,0,128)
RED      = (0, 0, 255)

# colors (RGB)
CYAN_RGB    = [0, 255, 255]
MAGENTA_RGB = [255, 0, 255]
RED_RGB     = [255, 0, 0]
GREEN_RGB   = [0, 255, 0]
BLUE_RGB    = [0, 0, 255]

DEFAULT_ALPHA = 0.3

def hu_to_grayscale(volume, hu_min, hu_max):
    # Clip at max and min values if specified
    if hu_min is not None or hu_max is not None:
        volume = np.clip(volume, hu_min, hu_max)

    # Scale to values between 0 and 1
    mxval = np.max(volume)
    mnval = np.min(volume)
    im_volume = (volume - mnval)/max(mxval - mnval, 1e-3)

    # Return values scaled to 0-255 range, but *not cast to uint8*
    # Repeat three times to make compatible with color overlay
    im_volume = 255*im_volume
    return np.stack((im_volume, im_volume, im_volume), axis=-1)


# 2d version
def draw_uni_contour(img_data, seg_data, pred_data=None):
    # img_data, seg_data, pred_data are all 2d images.
    # clip image and convert to BGR format.
    img_data = hu_to_grayscale(img_data, -76, 200)
    
    ret_s, thresh_s = cv2.threshold(seg_data, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours_s, _ = cv2.findContours(thresh_s, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    result = cv2.drawContours(img_data, contours_s, -1, CYAN, 1)

    if pred_data is not None:
        ret_p, thresh_p = cv2.threshold(pred_data, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours_p, _ = cv2.findContours(thresh_p, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        result = cv2.drawContours(result, contours_p, -1, RED, 1)
    
    return result


def draw_contour_volume(cid, pred=None, path='./pics'):
    img_data = load_volume(cid).get_fdata()
    seg_data = load_segmentation(cid).get_fdata()

    seg_data = seg_data.astype(np.uint8)
    pred = pred.astype(np.uint8)

    new_path = os.path.join(path, 'pics')
    os.mkdir(new_path)

    for i, sli in enumerate(img_data):
        if pred is not None:
            result = draw_uni_contour(img_data[i] , seg_data[i], pred[i])
        else:
            result = draw_uni_contour(img_data[i] , seg_data[i])

        pic_name = str(i).zfill(4) + '.png'
        cv2.imwrite(os.path.join(new_path, pic_name), result)


def class_to_color(segmentation, k_color, t_color):
    # initialize output to zeros
    shp = segmentation.shape
    seg_color = np.zeros((shp[0], shp[1], shp[2], 3), dtype=np.float32)

    # set output to appropriate color at each location
    seg_color[np.equal(segmentation, 1)] = k_color
    seg_color[np.equal(segmentation, 2)] = t_color
    return seg_color


def overlay(volume_ims, segmentation_ims, segmentation, alpha):
    # Get binary array for places where an ROI lives
    segbin = np.greater(segmentation, 0)
    repeated_segbin = np.stack((segbin, segbin, segbin), axis=-1)
    # Weighted sum where there's a value to overlay
    overlayed = np.where(
        repeated_segbin,
        np.round(alpha*segmentation_ims+(1-alpha)*volume_ims).astype(np.uint8),
        np.round(volume_ims).astype(np.uint8)
    )
    return overlayed


def visualize_patch(vol, seg, destination, pred=None, hu_min=-512, hu_max=512,
                    k_color=RED_RGB, t_color=BLUE_RGB, pk_color=MAGENTA_RGB,
                    pt_color=CYAN_RGB, alpha=DEFAULT_ALPHA, save=True):
    # Load segmentation and volume
    # vol = load_volume(cid)
    # vol = vol.get_data()
    
    seg = seg.astype(np.int32)

    # Convert to a visual format
    vol_ims = hu_to_grayscale(vol, hu_min, hu_max)
    # print(vol_ims.shape)
    seg_ims = class_to_color(seg, k_color, t_color)
    # print(seg_ims.shape)
    
    # Overlay the segmentation colors
    viz_ims = overlay(vol_ims, seg_ims, seg, alpha)
    # print(viz_ims.shape)
    if pred is not None:
        pred_ims = class_to_color(pred, pk_color, pt_color)
        viz_ims = overlay(viz_ims, pred_ims, pred, alpha)

    if save:
        # Prepare output location
        out_path = Path(destination)
        if not out_path.exists():
            out_path.mkdir()
        # Save individual images to disk
        for i in range(viz_ims.shape[0]):
            fpath = out_path / ("{:05d}.png".format(i))
            scipy.misc.imsave(str(fpath), viz_ims[i])

    return viz_ims