# Model 0: using Functional API to create a transfer learning model with 10% of data
# Start model_0.sh before running this script
# Author: Mark Minerov <markminerov123@gmail.com>

from helper_functions import create_tensorboard_callback, plot_loss_curves, unzip_data, walk_through_dir
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt

unzip_data("10_food_classes_10_percent.zip")
walk_through_dir("10_food_classes_10_percent")

train_dir = "10_food_classes_10_percent/train"
test_dir = "10_food_classes_10_percent/train"

IMG_SIZE = (224, 224)

train_data_10_percent = image_dataset_from_directory(
  directory=train_dir,
  image_size=IMG_SIZE,
  label_mode="categorical",
  batch_size=32
)

test_data = image_dataset_from_directory(
  directory=test_dir,
  image_size=IMG_SIZE,
  label_mode="categorical",
)

class_names = train_data_10_percent.class_names

base_model = tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False

inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")
x = base_model(inputs)

# for ResNet50V2
# x = tf.keras.layers.experemental.preprocessing.Rescaling(1/255.)(inputs)
# print(x.shape)

# extract features from each image, it is required by a transfer learning model:
# GlobalMaxPooling2D vs. GlobalAveragePooling2D

x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
outputs = tf.keras.layers.Dense(10, activation="softmax", name="output_layer")(x)

model_0 = tf.keras.Model(inputs, outputs)

model_0.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])

model_0_history = model_0.fit(
  train_data_10_percent,
  epochs=5,
  steps_per_epoch=len(train_data_10_percent),
  validation_data=test_data,
  validation_steps=len(test_data),
  callbacks=[create_tensorboard_callback(dir_name="transfer_learning", experiment_name="10_percent")]
)

model_0.evaluate(test_data)
plot_loss_curves(model_0_history)
plt.show()