# Model 4: Fine-tuning on existing model on all of the data
# Start model_4.sh before running this script
# Author: Mark Minerov <markminerov123@gmail.com>

from helper_functions import create_tensorboard_callback, plot_loss_curves, unzip_data, walk_through_dir
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers import GlobalAveragePooling2D
from utils import compare_histories

IMG_SIZE = (224, 224)
initial_epochs = 5
fine_tuning_epochs = initial_epochs + 5

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
  epochs=initial_epochs,
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

unzip_data("10_food_classes_all_data.zip")

# Setup training and test dirs
train_dir = "10_food_classes_all_data/train"
test_dir = "10_food_classes_all_data/test"

walk_through_dir("10_food_classes_all_data")

IMG_SIZE = (224, 224)

train_data_10_classes_full = tf.keras.preprocessing.image_dataset_from_directory(
  train_dir,
  label_mode="categorical",
  image_size=IMG_SIZE
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
  test_dir,
  label_mode="categorical",
  image_size=IMG_SIZE
)

model_2.compile(
  loss="categorical_crossentropy",
  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
  metrics=["accuracy"]
)

history_fine_10_classes_full = model_2.fit(
  train_data_10_classes_full,
  epochs=fine_tuning_epochs,
  validation_data=test_data,
  validation_steps=int(0.25 * len(test_data)),
  initial_epoch=model_2_history.epoch[-1],
  callbacks=[create_tensorboard_callback(
      dir_name="transfer_learning",
      experiment_name="full_10_classes_fine_tune_last_10"
  )]
)

model_2.evaluate(test_data)

compare_histories(model_2_history, history_fine_10_classes_full, initial_epochs=5)