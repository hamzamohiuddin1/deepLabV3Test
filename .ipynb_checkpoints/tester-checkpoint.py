import tensorflow as tf

# Check if TensorFlow is built with CUDA
print("CUDA Version:", tf.test.is_built_with_cuda())

# Check cuDNN version (if available)
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    # Assuming there is at least one GPU, check the cuDNN version for the first GPU
    gpu_properties = tf.config.experimental.get_device_details(gpus[0])
    print("cuDNN Version:", gpu_properties['device_lib']['cudnn_version'])
else:
    print("No GPU devices found.")