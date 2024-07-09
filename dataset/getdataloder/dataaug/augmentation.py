
import numpy as np
from batchgenerators.transforms.abstract_transforms import  Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform, ResizeTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, BrightnessTransform,ContrastAugmentationTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.transforms.abstract_transforms import AbstractTransform
import pickle

def load_from_pkl(load_path):
    data_input = open(load_path, 'rb')
    read_data = pickle.load(data_input)
    data_input.close()
    return read_data

class MaxMinNormalizationTransform(AbstractTransform):
    '''Rescales data into the specified range

    Args:
        rnge (tuple of float): The range to which the data is scaled

        per_channel (bool): determines whether the min and max values used for the rescaling are computed over the whole
        sample or separately for each channel

    '''

    def __init__(self, rnge=(0, 1), per_channel=True, data_key="data", label_key="seg"):
        self.data_key = data_key
        self.label_key = label_key
        self.per_channel = per_channel
        self.rnge = rnge

    def min_max_normalization(self,data, eps):
        mn = data.min()
        mx = data.max()
        data_normalized = data - mn
        old_range = mx - mn + eps
        data_normalized /= old_range

        return data_normalized
    def __call__(self, **data_dict):
        data_dict[self.data_key] = self.min_max_normalization(data_dict[self.data_key], eps=1e-8)
        return data_dict


def get_train_transform_2D(patch_size,
                           mirrir_p=0.5,gaussblur_p=0.2,
                           sharp_p=0.5, sharpen_l=10, sharpen_h=30,
                           gaussblur_l=0.25, gaussblur_h=1.5,
                           gaussnoise_p=0.15, gaussnoise_l=0, gaussnoise_h=1.0,
                           brightness_p=0.5, brightness_mean=0, brightness_std=0.1,
                           gamma_p=0.1, gamma_l=0.5, gamma_h=1,
                           perturbation_p=0.5, pertur_add_l=0, pertur_add_h=0.1, pertur_multi_l=0.9, pertur_multi_h=1.1,
                           elastic_p=0.5, elastic_alpha_l=0., elastic_alpha_h=1000., elastic_sigma_l=10., elastic_sigma_h=13.,
                           rotation_p=0.5, angle_xy=20,
                           scale_p=0.3, scale_l=0.4, scale_h=1.6): #rotation_p 0.2->0.5
    img_trans = {
        'train': Compose([MirrorTransform(p_per_sample=mirrir_p), #镜像
                          GaussianBlurTransform(blur_sigma=(0.5, 0.8), different_sigma_per_channel=True,
                                                p_per_channel=0.5, p_per_sample=gaussblur_p),
                          GaussianNoiseTransform(noise_variance=(0, 0.6), p_per_sample=gaussnoise_p,data_key="data"),
                          GammaTransform(gamma_range=(0.7, 1.5), invert_image=False, per_channel=True, p_per_sample=gamma_p),
                          SpatialTransform(patch_size, [i // 2 for i in patch_size],
                                           do_elastic_deform=False, alpha=(0., 900.), sigma=(9., 13.),  # 弹性变形
                                           do_rotation=True, angle_x=(-30. / 360. * 2 * np.pi, 30. / 360. * 2 * np.pi),
                                           # 这里斌哥改动了
                                           angle_y=(-15 / 360. * 2 * np.pi, -15. / 360. * 2 * np.pi),
                                           do_scale=True, scale=(0.85, 1.25),  # 缩放
                                           border_mode_data='constant', border_cval_data=0, order_data=3,  # 这个参数也和原始不一样
                                           border_mode_seg='constant', border_cval_seg=0, order_seg=0,  # seg推荐用0
                                           random_crop=False, data_key="data", label_key="seg",
                                           p_el_per_sample=0.0, p_scale_per_sample=scale_p, p_rot_per_sample=rotation_p),
                          ContrastAugmentationTransform(p_per_sample=0.15),
                          SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                         p_per_channel=0.5,
                                                         order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                         ignore_axes=None),
                          ResizeTransform(target_size=patch_size),
                          MaxMinNormalizationTransform(),
                          NumpyToTensor(['data', 'seg'], 'float')
                          ]),
        'val': Compose([ResizeTransform(target_size=patch_size)]),
        'seaug': Compose([ResizeTransform(target_size=patch_size),
                          SharpenTransform(magnitude=(sharpen_l, sharpen_h), p=sharp_p),
                          GaussianBlurTransform(blur_sigma=(gaussblur_l, gaussblur_h), p_per_sample=gaussblur_p),
                          GaussianNoiseTransform(noise_variance=(gaussnoise_l, gaussnoise_h), p_per_sample=gaussnoise_p),
                          BrightnessTransform(brightness_mean, brightness_std, per_channel=True,
                                              data_key="data", p_per_sample=brightness_p),
                          GammaTransform(gamma_range=(gamma_l, gamma_h), invert_image=False,
                                         per_channel=True, p_per_sample=gamma_p),
                          PerturbationTransform(add_magnitude=(pertur_add_l, pertur_add_h),
                                                multi_magnitude=(pertur_multi_l, pertur_multi_h), p=perturbation_p),
                          SpatialTransform(patch_size, [i // 2 for i in patch_size],
                                           do_elastic_deform=True, alpha=(elastic_alpha_l, elastic_alpha_h),
                                           sigma=(elastic_sigma_l, elastic_sigma_h),
                                           do_rotation=True,
                                           angle_x=(-1 * angle_xy / 360. * 2 * np.pi, angle_xy / 360. * 2 * np.pi),
                                           angle_y=(-1 * angle_xy / 360. * 2 * np.pi, angle_xy / 360. * 2 * np.pi),
                                           do_scale=True, scale=(scale_l, scale_h),
                                           border_mode_data='constant', border_cval_data=0, order_data=3,
                                           border_mode_seg='constant', border_cval_seg=0, order_seg=0,
                                           random_crop=True, data_key="data", label_key="seg",
                                           p_el_per_sample=elastic_p, p_scale_per_sample=scale_p, p_rot_per_sample=rotation_p),
                          ]),
    }

    return img_trans


class SharpenTransform(AbstractTransform):
    """Adds Sharpen

    """

    def __init__(self, magnitude=(10, 30), data_key="data", label_key="seg", p=0.5):
        self.blurr = GaussianBlurTransform(blur_sigma=(0.25, 1.5), different_sigma_per_channel=True,
                                           p_per_channel=0.5, p_per_sample=1)
        self.magnitude = magnitude
        self.p = p

    def __call__(self, **data_dict):
        mask = data_dict.get("seg")
        if np.random.uniform() < self.p:
            blurr_1 = self.blurr(**data_dict)
            blurr_2 = self.blurr(**blurr_1)
            blurr_1_image = blurr_1.get("data")
            blurr_2_image = blurr_2.get("data")
            alpha = np.random.uniform(self.magnitude[0], self.magnitude[1])
            sharpen_image = blurr_1_image + (blurr_1_image - blurr_2_image) * alpha
            data_dict = dict(data=sharpen_image, seg=mask)
        return data_dict

class PerturbationTransform(AbstractTransform):
    """Perturbation
    """
    def __init__(self, add_magnitude=(0, 0.1), multi_magnitude=(0.9, 1.1), data_key="data", label_key="seg", p=0.5):
        self.brightness = BrightnessTransform(add_magnitude[0], add_magnitude[1], p_per_sample=1)
        self.contrast = BrightnessMultiplicativeTransform(multi_magnitude, p_per_sample=1)
        self.p = p

    def __call__(self, **data_dict):
        if np.random.uniform() < self.p:
            data_dict = self.contrast(**data_dict)
            data_dict = self.brightness(**data_dict)
        return data_dict




