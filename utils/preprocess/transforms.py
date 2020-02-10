import numpy as np
from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform, SpatialTransform, ZoomTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from medicaltorch import transforms as mt_transforms
from medicaltorch import losses as mt_losses
from torchvision import transforms

train_transform = transforms.Compose([
        # mt_transforms.CenterCrop2D((200, 200)),
        mt_transforms.ElasticTransform(alpha_range=(28.0, 30.0),
                                       sigma_range=(3.5, 4.0),
                                       p=0.3),
        mt_transforms.RandomAffine(degrees=4.6,
                                   scale=(0.98, 1.02),
                                   translate=(0.03, 0.03)),
        mt_transforms.RandomTensorChannelShift((-0.10, 0.10)),
        mt_transforms.ToTensor()
        # mt_transforms.NormalizeInstance(),
    ])

gamma_t = GammaTransform(data_key="img", gamma_range=(0.1, 10))

mirror_t = MirrorTransform(data_key="img", label_key="seg")

spatial_t = SpatialTransform(patch_size=(8,8,8), data_key="img", label_key="seg")

gauss_noise_t = GaussianNoiseTransform(data_key="img", noise_variance=(0, 1))

zoom_t = ZoomTransform(zoom_factors=2, data_key="img")


def show_basic(x, gt, info=None):
    if info is not None:
        print("Test for " + info)

    print("img size: {}, max: {}, min: {}, avg: {}.".format(
        x.shape, np.max(x), np.min(x), np.average(x)))
    print("gt size: {}, max: {}, min: {}, avg: {}.\n".format(
        gt.shape, np.max(gt), np.min(gt), np.average(gt)))

if __name__ == "__main__":
    x = np.random.randn(1,2,32,32,32)
    gt = np.random.randn(1,2,32,32,32)
    # gt should be integers
    show_basic(x, gt, info="Original")
    
    y1 = gamma_t(img=x.copy())
    show_basic(y1['img'], gt, info="Gamma")
    y2 = mirror_t(img=x.copy(), seg=gt.copy())
    show_basic(y2['img'], y2['seg'], info="Mirror")
    show_basic(x, gt, info="Ori")
    y3 = spatial_t(img=x.copy(), seg=gt.copy())
    show_basic(y3['img'], y3['seg'], info="Spatial")
    y5 = gauss_noise_t(img=x.copy(), seg=gt.copy())
    show_basic(y5['img'], y5['seg'], info="Gaussian Noise")
    # y4 = zoom_t(img=x.copy(), seg=gt.copy())
    # show_basic(y4['img'], y4['seg'], info="Zoom")
