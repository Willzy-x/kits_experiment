import numpy as np
from scipy.ndimage import gaussian_filter
import skimage.measure as skmeasure
import scipy.ndimage as ndi
import torch

"""
Bbox and Landmark generation.
Note that all bbox and landmark returns (x, y, z)
But  the data is indexed by [z, y, x]. 
"""


class BBoxException(Exception):
    pass


# TODO : add func param to replace the mask != 0
def get_non_empty_center_idx_along_axis(mask, axis):
    """
    Get non zero center index along given axis.
    :param mask:
    :param axis:
    :return:
    """
    if isinstance(mask, torch.Tensor):
        # pytorch is the axis you want to get
        nonzero_idx = (mask != 0).nonzero()
        sorted_idx, _ = nonzero_idx[:, axis].sort()
        center_idx = sorted_idx[len(sorted_idx) // 2]
    elif isinstance(mask, np.ndarray):
        nonzero_idx = (mask != 0).nonzero()[axis]
        nonzero_idx.sort()
        center_idx = nonzero_idx[len(nonzero_idx) // 2]
    else:
        raise BBoxException("Wrong type")
    return center_idx


def get_non_empty_min_max_idx_along_axis(mask, axis):
    """
    Get non zero min and max index along given axis.
    :param mask:
    :param axis:
    :return:
    """
    if isinstance(mask, torch.Tensor):
        # pytorch is the axis you want to get
        nonzero_idx = (mask != 0).nonzero()
        if len(nonzero_idx) == 0:
            min = max = 0
        else:
            max = nonzero_idx[:, axis].max()
            min = nonzero_idx[:, axis].min()
    elif isinstance(mask, np.ndarray):
        nonzero_idx = (mask != 0).nonzero()
        if len(nonzero_idx[axis]) == 0:
            min = max = 0
        else:
            max = nonzero_idx[axis].max()
            min = nonzero_idx[axis].min()
    else:
        raise BBoxException("Wrong type")
    return min, max


def get_bbox_2d(mask):
    """ Input : [H, W], output : ((min_x, max_x), (min_y, max_y))
    Return non zero value's min and max index for a mask
    If no value exists, an array of all zero returns
    :param mask: numpy of [D, H, W]
    :return:
    """
    assert len(mask.shape) == 2
    min_y, max_y = get_non_empty_min_max_idx_along_axis(mask, 0)
    min_x, max_x = get_non_empty_min_max_idx_along_axis(mask, 1)
    return np.array(((min_x, max_x),
                     (min_y, max_y)))


def get_bbox_3d(mask):
    """ Input : [D, H, W] , output : ((min_x, max_x), (min_y, max_y), (min_z, max_z))
    Return non zero value's min and max index for a mask
    If no value exists, an array of all zero returns
    :param mask:  numpy of [D, H, W]
    :return:
    """
    assert len(mask.shape) == 3
    min_z, max_z = get_non_empty_min_max_idx_along_axis(mask, 0)
    min_y, max_y = get_non_empty_min_max_idx_along_axis(mask, 1)
    min_x, max_x = get_non_empty_min_max_idx_along_axis(mask, 2)
    return np.array(((min_x, max_x),
                     (min_y, max_y),
                     (min_z, max_z)))


def get_coords_from_corner(min_x, max_x, min_y, max_y):
    """
    Generate four corner coordinates from two corner coordinates.
    :param min_x:
    :param max_x:
    :param min_y:
    :param max_y:
    :return:
    """
    return [(min_x, min_y),
            (min_x, max_y),
            (max_x, max_y),
            (max_x, min_y)]


def generate_2d_landmark(slice, coords, sigma=3):
    H, W = slice.shape
    landmark_slices = np.zeros((5, H, W,), dtype=np.float)

    for coord in coords:
        for one_coord, slice in zip(coord, landmark_slices):
            slice[one_coord[1], one_coord[0]] = 1

    for i, slice in enumerate(landmark_slices):
        new_slice = gaussian_filter(slice, sigma, order=0)
        if new_slice.max() > 0:
            new_slice /= new_slice.max()
        landmark_slices[i] = new_slice
    return landmark_slices


def get_2d_landmarks_with_region(mask):
    """ Return left, up, right, bottom, center, coords indexed by (x, y)
    Note that the data is top-down reversed(along y axis, see test_bbox for detail)
    :param mask:
    :return:
    """
    # (left_x, right_x), (bottom_y, up_y) = get_bbox_2d(mask)

    tmplabel, numrg = ndi.measurements.label(mask)
    region_list = skmeasure.regionprops(tmplabel)

    coords = []

    if region_list == []:
        return []

    # assert len(region_list) <= 2

    for region in region_list[:2]:
        (up_y, left_x, bottom_y, right_x) = region.bbox
        # center landmark
        center_x, center_y = (left_x+right_x) // 2, (bottom_y+up_y)//2

        left_y = get_non_empty_center_idx_along_axis(mask[:, left_x], 0)
        right_y = get_non_empty_center_idx_along_axis(mask[:, right_x-1], 0)

        # find min index along specified axis
        up_x  = get_non_empty_center_idx_along_axis(mask[up_y, left_x:right_x], 0) + left_x
        bottom_x = get_non_empty_center_idx_along_axis(mask[bottom_y-1, left_x:right_x], 0) + left_x

        coord = np.array(((left_x, left_y),
                         (up_x, up_y),
                         (right_x, right_y),
                         (bottom_x, bottom_y),
                         (center_x, center_y)))
        coords.append(coord)
    return coords


def get_3d_landmarks(mask):
    """ Input : [D, H, W], output : ndarray of shape (6, 3),
    for (up, bottom, left, right, front, back, )
    :param mask:
    :return:
    """
    assert len(mask.shape) == 3

    def get_xy_coords(mask_slice):
        y = get_non_empty_center_idx_along_axis(mask_slice, 0)
        x = get_non_empty_center_idx_along_axis(mask_slice, 1)
        return x, y

    (left_x, right_x), (front_y, back_y), (bottom_z, up_z) = get_bbox_3d(mask)
    # [z, y, x]
    (left_z, left_y)   = get_xy_coords(mask[:, :, left_x])
    (right_z, right_y) = get_xy_coords(mask[:, :, right_x])
    (front_z, front_x) = get_xy_coords(mask[:, front_y, :])
    (back_z, back_x)   = get_xy_coords(mask[:, back_y, :])
    (bottom_y, bottom_x) = get_xy_coords(mask[bottom_z, :, :])
    (up_y, up_x)       = get_xy_coords(mask[up_z, :, :])

    landmarks = []
    landmarks.append((up_x, up_y, up_z))
    landmarks.append((bottom_x, bottom_y, bottom_z))
    landmarks.append((left_x, left_y, left_z))
    landmarks.append((right_x, right_y, right_z))
    landmarks.append((front_x, front_y, front_z))
    landmarks.append((back_x, back_y, back_z))
    return np.array(landmarks)


def get_bbox_from_3d_landmark(landmark, thresh=0.8, expand=[0, 0, 0], max_expand=None):
    """
    :param landmark : tensor of size [C, D, H, W]
    :return: a bounding box of size [3, 2]
    """
    if isinstance(landmark, np.ndarray):
        landmark = landmark.clip(0, 1)
    else:
        landmark = landmark.clamp(0, 1)
    # 0.2 for one path
    landmark[landmark < thresh] = 0
    bbox = get_bbox_3d(landmark)

    if max_expand is None:
        max_expand = expand
    expand = np.array(expand)
    bbox[:, 0] -= expand
    bbox[:, 1] += max_expand
    bbox = bbox.clip(0, 10000)
    return bbox


def get_local_maximum_2d(tensor):
    """
    :param tensor: tensor of size [D, H, W]
    :return: tensor : tensor of size [D, 2]
    """
    assert len(tensor.size())
    D, H, W = tensor.size()
    # get max indices from 1d array
    indices_1D = tensor.view(D, -1).argmax(1)
    # get real indices translated from 1d indices
    indices = torch.cat([(indices_1D / H).view(-1, 1), (indices_1D / W).view(-1, 1)], 1)
    return indices


def get_bbox_from_landmark_wich_local_max(landmark):
    """
    :param landmark : tensor of size [C, D, H, W]
    :return: a bounding box of size [3, 2]
    """
    left_indices = get_local_maximum_2d(landmark[0])
    left = left_indices[:, 0].argmin(0)

    up_indices = get_local_maximum_2d(landmark[1])
    up = up_indices[:, 1].argmin(0)

    right_indices = get_local_maximum_2d(landmark[2])
    right = right_indices[:, 0].argmax(0)

    bottom_indices = get_local_maximum_2d(landmark[3])
    bottom = bottom_indices[:, 1].argmax(0)
