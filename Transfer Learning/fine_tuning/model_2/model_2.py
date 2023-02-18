# Model 2: Feature extraction transfer learning model with 10% of data
# Start model_2.sh before running this script
# Author: Mark Minerov <markminerov123@gmail.com>

from helper_functions import create_tensorboard_callback, plot_loss_curves, unzip_data, walk_through_dir
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers import GlobalAveragePooling2D
import matplotlib.pyplot as plt
import numpy as np

IMG_SIZE = (224, 224)

unzip_data("10_food_classes_10_percent.zip")

train_dir_10_percent = "10_food_classes_10_percent/train"
test_dirt = "10_food_classes_10_percent/test"

train_data_10_percent = tf.keras.preprocessing.image_dataset_from_directory(
  train_dir_10_percent,
  label_mode="categorical",
  image_size=IMG_SIZE
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
  test_dirt,
  label_mode="categorical",
  image_size=IMG_SIZE
)

walk_through_dir("10_food_classes_10_percent")

data_augmentation = keras.Sequential([
  preprocessing.RandomFlip("horizontal"),
  preprocessing.RandomRotation(0.2),
  preprocessing.RandomZoom(0.2),
  preprocessing.RandomHeight(0.2),
  preprocessing.RandomWidth(0.2),
  # preprocessing.Rescaling(1./255) # keep for ResNet50V2, remove for EfficientNetB0
], name ="data_augmentation")


input_shape = IMG_SIZE + (3,)
base_model = tf.keras.applications.EfficientNetB0(include_top=False)
base_model.trainable = False

# 1. Create inputs layer
inputs = layers.Input(shape=input_shape, name="input_layer")

# 2. Create augmentation layer
x = data_augmentation(inputs)

# 3. Create base model layer
x = base_model(x, training=False)

# 4. Create global pooling layer for feature extraction
x = GlobalAveragePooling2D(name="global_average_pooling_2D")(x)

# 5. Create output layer
outputs = layers.Dense(10, activation="softmax", name="output_layer")(x)

model_2 = tf.keras.Model(inputs, outputs)

model_2.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(), metrics=["accuracy"])

# Set checkpoint path
checkpoint_path = "10_percent_model_checkpoints_weights/checkpoint.ckpt"

# Create a model checkpoint callback that saves the model's weights only
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
  filepath=checkpoint_path,
  save_weights_only=True,
  save_best_only=False,
  save_freq="epoch", # save every epoch
  verbose=1
)

model_2_history = model_2.fit(
  train_data_10_percent,
  epochs=5,
  steps_per_epoch=len(train_data_10_percent),
  validation_data=test_data,
  validation_steps=len(test_data),
  callbacks=[
    checkpoint_callback,
    create_tensorboard_callback(
      dir_name="transfer_learning",
      experiment_name="10_percent_data_aug")
    ]
)

results_10_percent_data_aug = model_2.evaluate(test_data)
plot_loss_curves(model_2_history)
plt.show()

# Load weights
model_2.load_weights(checkpoint_path)

# Evaluate our model on loaded results
loaded_weights_model_results = model_2.evaluate(test_data)

np.isclose(np.array(results_10_percent_data_aug), np.array(loaded_weights_model_results))