import tensorflow as tf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

print(tf.config.list_physical_devices(), tf.__version__)