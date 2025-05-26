import segmentation_models as sm
sm.set_framework('tf.keras')
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input
from tensorflow.keras.optimizers import Adam
from constants import *
import numpy as np

# from keras.optimizers import Adam

def get_pretrained_unet(input_shape=(512, 512, 3), backbone='resnet34', num_classes = 1, dropout=False):
    sm.set_framework('tf.keras')
    sm.framework()
    
    # use 'imagenet' weights, even if input is 1 channel (we'll adapt below)
    base_model = sm.Unet(
        backbone_name=backbone,
        input_shape=input_shape,  # pretrained models expect 3 channels
        classes=num_classes,
        activation='sigmoid',
        encoder_weights='imagenet'
    )
    
    # Inspect layer names if needed
    # for i, layer in enumerate(base_model.layers):
    #     print(i, layer.name, layer.output.shape)

    # Use decoder stage outputs as auxiliary prediction layers
    aux_1 = base_model.get_layer('decoder_stage2a_relu').output # lower resolution output (128, 128)
    aux_2 = base_model.get_layer('decoder_stage3a_relu').output # middle resolution output (256, 256)
    
    aux1_up = UpSampling2D(size=(4, 4), name="aux1_upsampling")(aux_1)
    aux1_out = Conv2D(1, (1, 1), activation='sigmoid', name="aux1_output")(aux1_up)
    
    aux2_up = UpSampling2D(size=(2, 2), name="aux2_upsampling")(aux_2)
    aux2_out = Conv2D(1, (1, 1), activation='sigmoid', name="aux2_output")(aux2_up)
    
    final_output = base_model.output # already sigmoid activated
    
    model = Model(inputs=base_model.input, outputs=[final_output, aux1_out, aux2_out])
    
    # If input images are grayscale, stack channels
    def preprocess(x):
        if x.shape[-1] == 1:
            return np.repeat(x, 3, axis=-1)
        return x

    model.compile(
        optimizer=Adam(1e-4),
        loss=[sm.losses.DiceLoss(), sm.losses.DiceLoss(), sm.losses.DiceLoss()],
        loss_weights=[0.6, 0.3, 0.1],
        metrics=[sm.metrics.IOUScore(threshold=0.5)]
    )

    print("Summary of pre-trained model", model.summary())  # Helpful for debugging --> prints model architecture
    
    # Return the model and preprocessing function
    return model, preprocess

def convert_grayscale_batch_to_rgb(x_batch):
    # Convert grayscale (N, H, W, 1) to RGB (N, H, W, 3)
    if x_batch.shape[-1] == 1:
        return np.repeat(x_batch, 3, axis=-1)
    return x_batch

def test_model_summary():
    model, _ = get_pretrained_unet()
    model.summary()

if __name__ == "__main__":
    test_model_summary()