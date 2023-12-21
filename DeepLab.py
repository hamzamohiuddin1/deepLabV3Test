from DeepLabV3Plus.deeplabv3plus.model.deeplabv3_plus import DeeplabV3Plus as DeepLabV3Plus
import tensorflow as tf

model = DeepLabV3Plus(backbone='resnet50', num_classes=20)
input_shape = (1, 512, 512, 3)
input_tensor = tf.random.normal(input_shape)
result = model(input_tensor)
model.summary()