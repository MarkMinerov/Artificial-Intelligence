# Model 1: 1% of data with data augmentation
# Start model_1.sh before running this script
# Author: Mark Minerov <markminerov123@gmail.com>

from helper_functions import create_tensorboard_callback, plot_loss_curves, unzip_data, walk_through_dir
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random
from tensorflow.keras.layers import GlobalAveragePooling2D

unzip_data("10_food_classes_1_percent.zip")

train_dir_1_percent = "10_food_classes_1_percent/train"
test_dir = "10_food_classes_1_percent/test"

walk_through_dir("10_food_classes_1_percent")

# Setup data loaders
IMG_SIZE = (224, 224)

train_data_1_percent = image_dataset_from_directory(
  train_dir_1_percent,
  label_mode="categorical",
  image_size=IMG_SIZE
)

test_data = image_dataset_from_directory(
  test_dir,
  label_mode="categorical",
  image_size=IMG_SIZE
)

# Data augmentation

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

data_augmentation = keras.Sequential([
  preprocessing.RandomFlip("horizontal"),
  preprocessing.RandomRotation(0.2),
  preprocessing.RandomZoom(0.2),
  preprocessing.RandomHeight(0.2),
  preprocessing.RandomWidth(0.2),
  # preprocessing.Rescaling(1./255) # keep for ResNet50V2, remove for EfficientNetB0
], name ="data_augmentation")

target_class = random.choice(train_data_1_percent.class_names)
target_dir = os.path.join("10_food_classes_1_percent/train/", target_class)
random_image = random.choice(os.listdir(target_dir))
random_image_path = os.path.join(target_dir, random_image)

img = mpimg.imread(random_image_path)
plt.imshow(img)
plt.axis(False)
plt.title(f"Original random image from class {target_class}");

augmented_img = data_augmentation(img)
plt.figure()
plt.imshow(tf.cast(augmented_img, dtype=tf.int32))
plt.axis(False)

plt.show()

base_model = tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False

input_shape = (224, 224, 3)

inputs = layers.Input(shape=input_shape, name="input_layer")

x = data_augmentation(inputs)
x = base_model(x, training=False)
x = GlobalAveragePooling2D(name="global_average_pooling_layer")(x)

# add Dense layer as the output

outputs = layers.Dense(10, activation="softmax", name="output_layer")(x)
model_1 = keras.Model(inputs, outputs)

model_1.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])

history_1_percent = model_1.fit(
  train_data_1_percent,
  epochs=5,
  steps_per_epoch=len(train_data_1_percent),
  validation_data=test_data,
  validation_steps=len(test_data),
  callbacks=[create_tensorboard_callback(dir_name="transfer_learning", experiment_name="1_percent_data_aug")]
)

model_1.evaluate(test_data)
plot_loss_curves(history_1_percent)