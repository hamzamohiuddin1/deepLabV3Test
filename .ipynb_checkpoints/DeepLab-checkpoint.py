from DeepLabV3Plus.deeplabv3plus.model.deeplabv3_plus import DeeplabV3Plus as DeepLabV3Plus
import numpy as np
from glob import glob
import tensorflow as tf
from random import shuffle
from DeepLabV3Plus.deeplabv3plus.train import Trainer
from DeepLabV3Plus.deeplabv3plus.inference import read_image, infer
from DeepLabV3Plus.deeplabv3plus.utils import (
    decode_segmask, get_overlay,
    plot_samples_matplotlib
)


model_file = '.DeepLabV3Plus/dataset/deeplabv3-plus-human-parsing-resnet-50-backbone.h5'
train_images = glob('./DeepLabV3Plus/dataset/instance-level_human_parsing/Training/Images/*')
val_images = glob('./DeepLabV3Plus/dataset/instance-level_human_parsing/Validation/Images/*')
test_images = glob('./DeepLabV3Plus/dataset/instance-level_human_parsing/Testing/Images/*')

def test_model():
    model = DeepLabV3Plus(backbone='resnet50', num_classes=20)
    input_shape = (1, 512, 512, 3)
    input_tensor = tf.random.normal(input_shape)
    result = model(input_tensor)
    model.summary()


def plot_predictions(images_list, size):
    for image_file in images_list:
        image_tensor = read_image(image_file, size)
        prediction = infer(
            image_tensor=image_tensor,
            model_file=model_file
        )
        plot_samples_matplotlib(
            [image_tensor, prediction], figsize=(10, 6)
        )


def main():
    plot_predictions(train_images[:4], (512, 512))

if __name__=="__main__":
    main()