import numpy as np
import random
import torch
from PIL import Image
from torchvision.transforms import (
    Resize, Compose, ToTensor, Normalize, CenterCrop, RandomCrop,
    RandomResizedCrop, RandomHorizontalFlip, Grayscale, Lambda
)

from .autoaugment import SVHNPolicy, CIFAR10Policy, ImageNetPolicy
from .randaugment import RandAugment, RandAugment2, RandAugmentDuckie, RandAugmentFixMatch

AVAI_CHOICES = [
    'random_flip', 'random_resized_crop', 'normalize', 'instance_norm',
    'random_crop', 'random_translation', 'center_crop', 'cutout',
    'imagenet_policy', 'cifar10_policy', 'svhn_policy', 'randaugment',
    'randaugment_fixmatch', 'randaugment2', 'randaugment_duckie', 'gaussian_noise',
    'zero_center', 'grayscale'
]


class Random2DTranslation:
    """Randomly translates the input image with a probability.
    Specifically, given a predefined shape (height, width), the
    input is first resized with a factor of 1.125, leading to
    (height*1.125, width*1.125), then a random crop is performed.
    Such operation is done with a probability.

    Args:
        height (int): target image height.
        width (int): target image width.
        p (float, optional): probability that this operation takes place.
            Default is 0.5.
        interpolation (int, optional): desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img.resize((self.width, self.height), self.interpolation)

        new_width = int(round(self.width * 1.125))
        new_height = int(round(self.height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)

        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop(
            (x1, y1, x1 + self.width, y1 + self.height)
        )

        return croped_img


class InstanceNormalization:
    """Normalize data using per-channel mean and standard deviation.

    Reference:
        - Ulyanov et al. Instance normalization: The missing in- gredient
          for fast stylization. ArXiv 2016.
        - Shu et al. A DIRT-T Approach to Unsupervised Domain Adaptation.
          ICLR 2018.
    """

    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, img):
        C, H, W = img.shape
        img_re = img.reshape(C, H * W)
        mean = img_re.mean(1).view(C, 1, 1)
        std = img_re.std(1).view(C, 1, 1)
        return (img-mean) / (std + self.eps)


class Cutout:
    """Randomly mask out one or more patches from an image.

    https://github.com/uoguelph-mlrg/Cutout

    Args:
        n_holes (int, optional): number of patches to cut out
            of each image. Default is 1.
        length (int, optinal): length (in pixels) of each square
            patch. Default is 16.
    """

    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): tensor image of size (C, H, W).

        Returns:
            Tensor: image with n_holes of dimension
                length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        return img * mask


class GaussianNoise:
    """Add gaussian noise."""

    def __init__(self, mean=0, std=0.15, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img
        noise = torch.randn(img.size()) * self.std + self.mean
        return img + noise

class ZeroCenter():
    def __init__(self):
        self.zero_center = Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))
    def __call__(self, tensor):
        return self.zero_center(tensor)
    
def build_transform(cfg, is_train=True, choices=None):
    """Build transformation function.

    Args:
        cfg (CfgNode): config.
        is_train (bool, optional): for training (True) or test (False).
            Default is True.
        choices (list, optional): list of strings which will overwrite
            cfg.INPUT.TRANSFORMS if given. Default is None.
    """
    if cfg.INPUT.NO_TRANSFORM:
        print('Note: no transform is applied!')
        return None

    if choices is None:
        choices = cfg.INPUT.TRANSFORMS

    for choice in choices:
        assert choice in AVAI_CHOICES, \
            'Invalid transform choice ({}), ' \
            'expected to be one of {}'.format(choice, AVAI_CHOICES)

    expected_size = '{}x{}'.format(cfg.INPUT.SIZE[0], cfg.INPUT.SIZE[1])

    normalize = Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)

    if is_train:
        return _build_transform_train(cfg, choices, expected_size, normalize)
    else:
        return _build_transform_test(cfg, choices, expected_size, normalize)


def _build_transform_train(cfg, choices, expected_size, normalize):
    print('Building transform_train')
    tfm_train = []
    
    print('+ resize to {}'.format(expected_size))
    tfm_train += [Resize(cfg.INPUT.SIZE)]
    
    if 'grayscale' in choices:
        print('+ grayscale')
        tfm_train += [Grayscale()]
    
    if 'random_flip' in choices:
        print('+ random flip')
        tfm_train += [RandomHorizontalFlip()]

    if 'random_translation' in choices:
        print('+ random translation')
        tfm_train += [
            Random2DTranslation(cfg.INPUT.SIZE[0], cfg.INPUT.SIZE[1])
        ]

    if 'random_crop' in choices:
        crop_padding = cfg.INPUT.CROP_PADDING
        print('+ random crop (padding = {})'.format(crop_padding))
        tfm_train += [RandomCrop(cfg.INPUT.SIZE, padding=crop_padding)]

    if 'random_resized_crop' in choices:
        print('+ random resized crop')
        tfm_train += [RandomResizedCrop(cfg.INPUT.SIZE)]

    if 'center_crop' in choices:
        print('+ center crop (on 1.125x enlarged input)')
        enlarged_size = [int(x * 1.125) for x in cfg.INPUT.SIZE]
        tfm_train += [Resize(enlarged_size)]
        tfm_train += [CenterCrop(cfg.INPUT.SIZE)]

    if 'imagenet_policy' in choices:
        print('+ imagenet policy')
        tfm_train += [ImageNetPolicy()]

    if 'cifar10_policy' in choices:
        print('+ cifar10 policy')
        tfm_train += [CIFAR10Policy()]

    if 'svhn_policy' in choices:
        print('+ svhn policy')
        tfm_train += [SVHNPolicy()]

    if 'randaugment' in choices:
        n_ = cfg.INPUT.RANDAUGMENT_N
        m_ = cfg.INPUT.RANDAUGMENT_M
        print('+ randaugment (n={}, m={})'.format(n_, m_))
        tfm_train += [RandAugment(n_, m_)]

    if 'randaugment_fixmatch' in choices:
        n_ = cfg.INPUT.RANDAUGMENT_N
        print('+ randaugment_fixmatch (n={})'.format(n_))
        tfm_train += [RandAugmentFixMatch(n_)]

    if 'randaugment2' in choices:
        n_ = cfg.INPUT.RANDAUGMENT_N
        print('+ randaugment2 (n={})'.format(n_))
        tfm_train += [RandAugment2(n_)]
    
    if 'randaugmentduckie' in choices:
        n_ = cfg.INPUT.RANDAUGMENT_N
        print('+ randaugmentduckie (n={})'.format(n_))
        tfm_train += [RandAugmentDuckie(n_)]

    print('+ to torch tensor of from range [0,255] to range [0, 1]')
    tfm_train += [ToTensor()]

    if 'cutout' in choices:
        cutout_n = cfg.INPUT.CUTOUT_N
        cutout_len = cfg.INPUT.CUTOUT_LEN
        print('+ cutout (n_holes={}, length={})'.format(cutout_n, cutout_len))
        tfm_train += [Cutout(cutout_n, cutout_len)]
        
    if 'zero_center' in choices:
        print('+ zero_center from range [min, max] to [0,1]')
        tfm_train += [ZeroCenter()]
        
    if 'normalize' in choices:
        print(
            '+ normalization (mean={}, '
            'std={})'.format(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)
        )
        tfm_train += [normalize]

    if 'gaussian_noise' in choices:
        print(
            '+ gaussian noise (mean={}, std={})'.format(
                cfg.INPUT.GN_MEAN, cfg.INPUT.GN_STD
            )
        )
        tfm_train += [GaussianNoise(cfg.INPUT.GN_MEAN, cfg.INPUT.GN_STD)]
    if 'instance_norm' in choices:
        print('+ instance normalization')
        tfm_train += [InstanceNormalization()]

    tfm_train = Compose(tfm_train)

    return tfm_train


def _build_transform_test(cfg, choices, expected_size, normalize):
    print('Building transform_test')
    tfm_test = []

    print('+ resize to {}'.format(expected_size))
    tfm_test += [Resize(cfg.INPUT.SIZE)]

    if 'center_crop' in choices:
        print('+ center crop (on 1.125x enlarged input)')
        enlarged_size = [int(x * 1.125) for x in cfg.INPUT.SIZE]
        tfm_test += [Resize(enlarged_size)]
        tfm_test += [CenterCrop(cfg.INPUT.SIZE)]
    if 'grayscale' in choices:
        print('+ grayscale')
        tfm_test += [Grayscale()]
    
    print('+ to torch tensor from range [0,255] to range [0, 1]')
    tfm_test += [ToTensor()]
    if 'zero_center' in choices:
        print('+ zero_center from range [min, max] to [0,1]')
        tfm_test += [ZeroCenter()]
        
    if 'normalize' in choices:
        print(
            '+ normalization (mean={}, '
            'std={})'.format(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)
        )
        tfm_test += [normalize]
    
    if 'instance_norm' in choices:
        print('+ instance normalization')
        tfm_test += [InstanceNormalization()]

    tfm_test = Compose(tfm_test)

    return tfm_test
