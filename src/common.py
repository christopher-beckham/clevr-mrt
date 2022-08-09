import os
from torch.utils.data import (DataLoader,
                              Subset)
from .iterators.datasets import (ClevrDataset,
                                 ClevrKiwiDataset,
                                 ClevrKiwiAutoencoderDataset,
                                 BlankDataset)
import torch
import numpy as np
from torchvision.transforms import transforms

from .setup_logger import get_logger
logger = get_logger()

# https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745
class AddGaussianNoise(object):
    def __init__(self, std_range=(0.0, 0.1)):
        self.std_range = std_range
        self.mean = 0.
    def __call__(self, tensor):
        # Sample a random sdev.
        std = np.random.uniform(self.std_range[0],
                                self.std_range[1])
        return tensor + torch.randn(tensor.size())*std
    def __repr__(self):
        return self.__class__.__name__

def get_color_distortion(s=0.5):
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    #rnd_gray = transforms.RandomGrayscale(p=0.2)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([ rnd_color_jitter, rnd_gray])
    return color_distort

def load_dataset(name,
                 img_size,
                 imagenet_scaling=False,
                 train=True,
                 **kwargs):

    if imagenet_scaling:
        norm_ = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.224))
    else:
        norm_ = transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                     std=(0.5, 0.5, 0.5))

    if name == 'clevr':
        if kwargs != {}:
            raise Exception("This dataset does not support extra dataset kwargs")
        # https://github.com/ethanjperez/film/blob/master/scripts/extract_features.py#L58
        train_transforms = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            norm_
        ]
        if train:
            ds_train = ClevrDataset(root_images=os.environ['DATASET_CLEVR_IMAGES'],
                                    root_meta=os.environ['DATASET_CLEVR_META'],
                                    transforms_=train_transforms,
                                    mode='train')
            ds_valid = ClevrDataset(root_images=os.environ['DATASET_CLEVR_IMAGES'],
                                    root_meta=os.environ['DATASET_CLEVR_META'],
                                    transforms_=train_transforms,
                                    mode='val')
        else:
            raise Exception("test set not yet added for clevr")
    elif name in ['clevr_kiwi', 'clevr_kiwi_cc', 'clevr_kiwi_nocc']:
        if kwargs != {}:
            raise Exception("This dataset does not support extra dataset kwargs")
        # https://github.com/ethanjperez/film/blob/master/scripts/extract_features.py#L58
        train_transforms = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            norm_
        ]
        
        logger.info("Transform: {}".format(train_transforms))

        canonical_mode = True if '_cc' in name else False

        if name == 'clevr_kiwi':
            # Only have it in train, don't
            # use cc in valid.
            canonical_mode = 'train_only'
        elif name == 'clevr_kiwi_cc':
            # Canonical view ONLY in train and valid
            # ('baseline 0', basically emulate
            # vanilla clevr)
            canonical_mode = 'b0'
        elif name == 'clevr_kiwi_nocc':
            # No canonical views used in train nor valid.
            # Should be the hardest option.
            canonical_mode = 'none'
        else:
            raise Exception("")

        if train:
            ds_train = ClevrKiwiDataset(
                root_images=os.environ['DATASET_CLEVR_KIWI'],
                root_meta=os.environ['DATASET_CLEVR_KIWI_META'],
                transforms_=train_transforms,
                canonical_mode=canonical_mode,
                mode='train'
            )
            ds_valid = ClevrKiwiDataset(
                root_images=os.environ['DATASET_CLEVR_KIWI'],
                root_meta=os.environ['DATASET_CLEVR_KIWI_META'],
                transforms_=train_transforms,
                canonical_mode=canonical_mode,
                mode='val',
                # Using the id_to_scene generated from ds_train so we
                # don't load in the cache file twice from disk
                id_to_scene=ds_train.id_to_scene
            )
        else:
            ds_test = ClevrKiwiDataset(
                root_images=os.environ['DATASET_CLEVR_KIWI_TEST'],
                root_meta=os.environ['DATASET_CLEVR_KIWI_META'],
                transforms_=train_transforms,
                canonical_mode=canonical_mode,
                mode='test'
            )
    elif name in ['clevr_kiwi_ae', 'clevr_kiwi_ae_da']:

        if name == 'clevr_kiwi_ae':
            if kwargs != {}:
                raise Exception("This dataset does not support dataset kwargs")
            train_transforms = [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                norm_
            ]
        else:
            logger.info("Dataset args: " + str(kwargs))
            min_scale = kwargs.pop('min_scale', 0.5)
            colour_strength = kwargs.pop('colour_strength', 0.)
            # `da_transforms` is what we use when we're
            # applying the contrastive loss.
            da_transforms = [
                transforms.RandomResizedCrop((img_size, img_size),
                                             scale=(min_scale, 1.0))
            ]
            if colour_strength > 0:
                da_transforms.append(
                    get_color_distortion(colour_strength)
                )
            da_transforms += [transforms.ToTensor(), norm_]
            # `blank_transforms` is what we use when we are
            # enforcing rotational consistency between
            # volumes (either in sup or unsup case).
            blank_transforms = [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                norm_
            ]
            logger.info("DA transforms (ONLY for train set): " + \
                        str(da_transforms))
            logger.info("Blank transforms (ONLY for valid set): " + \
                        str(blank_transforms))

        if train:
            ds_train = ClevrKiwiAutoencoderDataset(root_images=os.environ['DATASET_CLEVR_KIWI'],
                                                   root_meta=os.environ['DATASET_CLEVR_KIWI_META'],
                                                   transforms_=da_transforms,
                                                   transforms_blank=blank_transforms,
                                                   canonical_mode='train_only',
                                                   mode='train')
            ds_valid = ClevrKiwiAutoencoderDataset(root_images=os.environ['DATASET_CLEVR_KIWI'],
                                                   root_meta=os.environ['DATASET_CLEVR_KIWI_META'],
                                                   transforms_=blank_transforms,
                                                   transforms_blank=blank_transforms,
                                                   canonical_mode='train_only',
                                                   mode='val')
        else:
            ds_test = ClevrKiwiAutoencoderDataset(root_images=os.environ['DATASET_CLEVR_KIWI_TEST'],
                                                  root_meta=os.environ['DATASET_CLEVR_KIWI_META'],
                                                  transforms_=blank_transforms,
                                                  transforms_blank=blank_transforms,
                                                  canonical_mode='train_only',
                                                  mode='test')

    elif name == 'blank':

        from .iterators.utils import load_vocab
        vocab = load_vocab(os.environ["DATASET_CLEVR_KIWI_META"] +
                           "/vocab.json")
        if train:
            ds_train = BlankDataset()
            ds_valid = BlankDataset()
            ds_train.vocab = vocab
            ds_valid.vocab = vocab
        else:
            ds_test = BlankDataset()
            ds_test.vocab = vocab

    else:
        raise Exception("Specified dataset %s is not valid" % name)

    if train:
        return ds_train, ds_valid
    else:
        return ds_test
