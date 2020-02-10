from medicaltorch import transforms as mt_transforms
from medicaltorch import losses as mt_losses
from torchvision import transforms

packed_transforms = [
    mt_transforms.RandomRotation(degrees=(90, 180)),
    mt_transforms.ElasticTransform(),
    mt_transforms.AdditiveGaussianNoise(mean=0.0, std=0.05),
    mt_transforms.RandomAffine(),
    mt_transforms.ToTensor()
]
