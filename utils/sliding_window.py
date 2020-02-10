import numpy as np
from itertools import product
from numpy.lib.stride_tricks import as_strided

EPSILON = 1e-5


#
def reconstruct_labels(patches, image_size, nclass, extraction_step=1, m_patches=None):
    """
    In case of multiple occurrences of the maximum values, the indices corresponding to the first occurrence are returned.
    Integers only.
    :rtype: object
    """
    if isinstance(extraction_step, int):
        extraction_step = (extraction_step, extraction_step, extraction_step)
    i_h, i_w, i_d = image_size
    p_h, p_w, p_d = patches.shape[3:]
    img = np.zeros(image_size + (nclass,))

    for (i, j, k) in product(range(patches.shape[0]), range(patches.shape[1]), range(patches.shape[2])):
        loc_i = i * extraction_step[0]
        loc_j = j * extraction_step[1]
        loc_k = k * extraction_step[2]
        end_i = loc_i + p_h
        end_j = loc_j + p_w
        end_k = loc_k + p_d
        for label in range(nclass):
            img[loc_i:end_i, loc_j:end_j, loc_k:end_k, label][patches[i, j, k, :, :, :] == label] += 1

    if m_patches is not None:
        for (i, j, k) in product(range(m_patches.shape[0]), range(m_patches.shape[1]), range(m_patches.shape[2])):
            end_i = i_h - i * extraction_step[0]
            end_j = i_w - j * extraction_step[1]
            end_k = i_d - k * extraction_step[2]
            loc_i = end_i - p_h
            loc_j = end_j - p_w
            loc_k = end_k - p_d
            for label in range(nclass):
                img[loc_i:end_i, loc_j:end_j, loc_k:end_k, label][np.flip(
                    m_patches[i, j, k, :, :, :], axis=(0, 1, 2)) == label] += 1

    img = np.argmax(img, axis=-1)
    print(np.unique(img))
    return img


def reconstruct_patches(patches, image_size, extraction_step=1, m_patches=None):
    if isinstance(extraction_step, int):
        extraction_step = (extraction_step, extraction_step, extraction_step)
    i_h, i_w, i_d = image_size
    p_h, p_w, p_d = patches.shape[3:]
    img = np.zeros(image_size)
    count = np.zeros(image_size)
    count += EPSILON
    for (i, j, k) in product(range(patches.shape[0]), range(patches.shape[1]), range(patches.shape[2])):
        loc_i = i * extraction_step[0]
        loc_j = j * extraction_step[1]
        loc_k = k * extraction_step[2]
        end_i = loc_i + p_h
        end_j = loc_j + p_w
        end_k = loc_k + p_d
        img[loc_i:end_i, loc_j:end_j, loc_k:end_k] += patches[i, j, k, :, :, :]
        count[loc_i:end_i, loc_j:end_j, loc_k:end_k] += 1

    if m_patches is not None:
        for (i, j, k) in product(range(m_patches.shape[0]), range(m_patches.shape[1]), range(m_patches.shape[2])):
            end_i = i_h - i * extraction_step[0]
            end_j = i_w - j * extraction_step[1]
            end_k = i_d - k * extraction_step[2]
            loc_i = end_i - p_h
            loc_j = end_j - p_w
            loc_k = end_k - p_d
            img[loc_i:end_i, loc_j:end_j, loc_k:end_k] += np.flip(
                m_patches[i, j, k, :, :, :], axis=(0, 1, 2))
            count[loc_i:end_i, loc_j:end_j, loc_k:end_k] += 1

    img = img / count
    return img


def mirror_img(sample_data, axes=(2, 1, 0)):
    mirrored_axes = [0, 0, 0]
    vol = sample_data.copy()
    vol = np.transpose(vol, axes)
    return vol
