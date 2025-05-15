from constants import *
import numpy as np
import segmentation_models as sm
from keras.optimizers import Adam

def get_pretrained_unet(input_shape=(512, 512, 1), backbone='resnet34', dropout=False):
    sm.set_framework('tf.keras')
    sm.framework()
    
    # use 'imagenet' weights, even if input is 1 channel (we'll adapt below)
    base_model = sm.Unet(
        backbone_name=backbone,
        input_shape=(512, 512, 3),  # pretrained models expect 3 channels
        classes=1,
        activation='sigmoid',
        encoder_weights='imagenet'
    )
    
    # If input images are grayscale, stack channels
    def preprocess(x):
        if x.shape[-1] == 1:
            return np.repeat(x, 3, axis=-1)
        return x

    base_model.compile(
        optimizer=Adam(1e-4),
        loss=sm.losses.DiceLoss(),
        metrics=[sm.metrics.IOUScore(threshold=0.5)]
    )

    return base_model, preprocess

def convert_grayscale_batch_to_rgb(x_batch):
    # Convert grayscale (N, H, W, 1) to RGB (N, H, W, 3)
    if x_batch.shape[-1] == 1:
        return np.repeat(x_batch, 3, axis=-1)
    return x_batch