from copy import deepcopy
import numpy as np
from batchgenerators.dataloading import MultiThreadedAugmenter, SingleThreadedAugmenter
from batchgenerators.transforms import Compose, RenameTransform, GammaTransform, SpatialTransform
from batchgenerators.transforms import DataChannelSelectionTransform, SegChannelSelectionTransform
from batchgenerators.transforms import MirrorTransform, NumpyToTensor
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform
# from nnunet.training.data_augmentation.pyramid_augmentations import MoveSegAsOneHotToData, \
#     RemoveRandomConnectedComponentFromOneHotEncodingTransform, ApplyRandomBinaryOperatorTransform
# from nnunet.training.data_augmentation.custom_transforms import Convert3DTo2DTransform, Convert2DTo3DTransform, \
#     MaskTransform


default_3D_augmentation_params = {
    "selected_data_channels": None,
    "selected_seg_channels": None,
    "do_elastic": True,
    "elastic_deform_alpha": (0., 900.),
    "elastic_deform_sigma": (9., 13.),
    "do_scaling": True,
    "scale_range": (0.85, 1.25),
    "do_rotation": True,
    "rotation_x": (-15./360 * 2. * np.pi, 15./360 * 2. * np.pi),
    "rotation_y": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "rotation_z": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "random_crop": False,
    "random_crop_dist_to_border": None,
    "do_gamma": True,
    "gamma_retain_stats": True,
    "gamma_range": (0.7, 1.5),
    "p_gamma": 0.3,
    "num_threads": 8,
    "num_cached_per_thread": 1,
    "mirror": True,
    "mirror_axes": (0, 1, 2),
    "p_eldef": 0.2,
    "p_scale": 0.2,
    "p_rot": 0.2,
    "dummy_2D": False,
    "mask_was_used_for_normalization": False,
    "all_segmentation_labels": None,  # used for pyramid
    "move_last_seg_chanel_to_data": False,  # used for pyramid
    "border_mode_data": "constant",
    "advanced_pyramid_augmentations": False  # used for pyramid
}


def get_basic_patch_size(final_patch_size):
        basic_patch_size = get_patch_size(final_patch_size,
                                          default_3D_augmentation_params['rotation_x'],
                                          default_3D_augmentation_params['rotation_y'],
                                          default_3D_augmentation_params['rotation_z'],
                                          default_3D_augmentation_params['scale_range'])
        return basic_patch_size


def get_patch_size(final_patch_size, rot_x, rot_y, rot_z, scale_range):
    if isinstance(rot_x, (tuple, list)):
        rot_x = max(np.abs(rot_x))
    if isinstance(rot_y, (tuple, list)):
        rot_y = max(np.abs(rot_y))
    if isinstance(rot_z, (tuple, list)):
        rot_z = max(np.abs(rot_z))
    rot_x = min(90/360 * 2. * np.pi, rot_x)
    rot_y = min(90/360 * 2. * np.pi, rot_y)
    rot_z = min(90/360 * 2. * np.pi, rot_z)
    from batchgenerators.augmentations.utils import rotate_coords_3d, rotate_coords_2d
    coords = np.array(final_patch_size)
    final_shape = np.copy(coords)
    if len(coords) == 3:
        final_shape = np.max(
            np.vstack((rotate_coords_3d(coords, rot_x, 0, 0), final_shape)), 0)
        final_shape = np.max(
            np.vstack((rotate_coords_3d(coords, 0, rot_y, 0), final_shape)), 0)
        final_shape = np.max(
            np.vstack((rotate_coords_3d(coords, 0, 0, rot_z), final_shape)), 0)
    elif len(coords) == 2:
        final_shape = np.max(
            np.vstack((rotate_coords_2d(coords, rot_x), final_shape)), 0)
    final_shape /= min(scale_range)
    return final_shape.astype(int)
    

def get_default_augmentation(dataloader_train, dataloader_val, patch_size, params=default_3D_augmentation_params, border_val_seg=-1, pin_memory=True,
                             seeds_train=None, seeds_val=None):
    tr_transforms = []

    if params.get("selected_data_channels") is not None:
        tr_transforms.append(DataChannelSelectionTransform(
            params.get("selected_data_channels")))

    if params.get("selected_seg_channels") is not None:
        tr_transforms.append(SegChannelSelectionTransform(
            params.get("selected_seg_channels")))

    # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
    # if params.get("dummy_2D") is not None and params.get("dummy_2D"):
    #     tr_transforms.append(Convert3DTo2DTransform())

    tr_transforms.append(SpatialTransform(
        patch_size, patch_center_dist_from_border=None, do_elastic_deform=params.get("do_elastic"),
        alpha=params.get("elastic_deform_alpha"), sigma=params.get("elastic_deform_sigma"),
        do_rotation=params.get("do_rotation"), angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
        angle_z=params.get("rotation_z"), do_scale=params.get("do_scaling"), scale=params.get("scale_range"),
        border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=3, border_mode_seg="constant", border_cval_seg=border_val_seg,
        order_seg=1, random_crop=params.get("random_crop"), p_el_per_sample=params.get("p_eldef"),
        p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot")
    ))
    # if params.get("dummy_2D") is not None and params.get("dummy_2D"):
    #     tr_transforms.append(Convert2DTo3DTransform())

    if params.get("do_gamma"):
        tr_transforms.append(GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get(
            "gamma_retain_stats"), p_per_sample=params["p_gamma"]))

    tr_transforms.append(MirrorTransform(params.get("mirror_axes")))

    # if params.get("mask_was_used_for_normalization") is not None:
    #     mask_was_used_for_normalization = params.get(
    #         "mask_was_used_for_normalization")
    #     tr_transforms.append(MaskTransform(
    #         mask_was_used_for_normalization, mask_idx_in_seg=0, set_outside_to=0))

    tr_transforms.append(RemoveLabelTransform(-1, 0))

    # if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
    #     tr_transforms.append(MoveSegAsOneHotToData(
    #         1, params.get("all_segmentation_labels"), 'seg', 'data'))
    #     if params.get("advanced_pyramid_augmentations") and not None and params.get("advanced_pyramid_augmentations"):
    #         tr_transforms.append(ApplyRandomBinaryOperatorTransform(channel_idx=list(range(-len(params.get("all_segmentation_labels")), 0)),
    #                                                                 p_per_sample=0.4,
    #                                                                 key="data",
    #                                                                 strel_size=(1, 8)))
    #         tr_transforms.append(RemoveRandomConnectedComponentFromOneHotEncodingTransform(channel_idx=list(range(-len(params.get("all_segmentation_labels")), 0)),
    #                                                                                        key="data",
    #                                                                                        p_per_sample=0.2,
    #                                                                                        fill_with_other_class_p=0.0,
    #                                                                                        dont_do_if_covers_more_than_X_percent=0.15))

    tr_transforms.append(RenameTransform('seg', 'target', True))
    tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    tr_transforms = Compose(tr_transforms)
    #from batchgenerators.dataloading import SingleThreadedAugmenter
    # batchgenerator_train = SingleThreadedAugmenter(dataloader_train, tr_transforms)
    #import IPython;IPython.embed()

    batchgenerator_train = MultiThreadedAugmenter(dataloader_train, tr_transforms, params.get(
        'num_threads'), params.get("num_cached_per_thread"), seeds=seeds_train, pin_memory=pin_memory)

    val_transforms = []
    val_transforms.append(RemoveLabelTransform(-1, 0))
    if params.get("selected_data_channels") is not None:
        val_transforms.append(DataChannelSelectionTransform(
            params.get("selected_data_channels")))
    if params.get("selected_seg_channels") is not None:
        val_transforms.append(SegChannelSelectionTransform(
            params.get("selected_seg_channels")))

    # if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
    #     val_transforms.append(MoveSegAsOneHotToData(
    #         1, params.get("all_segmentation_labels"), 'seg', 'data'))

    val_transforms.append(RenameTransform('seg', 'target', True))
    val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    val_transforms = Compose(val_transforms)

    # batchgenerator_val = SingleThreadedAugmenter(dataloader_val, val_transforms)
    batchgenerator_val = MultiThreadedAugmenter(dataloader_val, val_transforms, max(params.get(
       'num_threads')//2, 1), params.get("num_cached_per_thread"), seeds=seeds_val, pin_memory=pin_memory)
    return batchgenerator_train, batchgenerator_val


    
