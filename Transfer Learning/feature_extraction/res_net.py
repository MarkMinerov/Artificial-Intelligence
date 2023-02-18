# !wget https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_10_percent.zip

import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from utils import create_model, create_tensorboard_callback, pred_and_plot, plot_loss
import pathlib
import numpy as np

zip_ref = zipfile.ZipFile("10_food_classes_10_percent.zip")
zip_ref.extractall()
zip_ref.close()

IMAGE_SHAPE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
resnet_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"

data_dir = pathlib.Path("10_food_classes_10_percent/train")
class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))

train_dir = "10_food_classes_10_percent/train/"
test_dir = "10_food_classes_10_percent/test/"

train_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = ImageDataGenerator(rescale=1/255.)

train_data_10_percent = train_datagen.flow_from_directory(
  train_dir,
  target_size=IMAGE_SHAPE,
  batch_size=BATCH_SIZE,
  class_mode="categorical"
)

test_data = test_datagen.flow_from_directory(
  test_dir,
  target_size=IMAGE_SHAPE,
  batch_size=BATCH_SIZE,
  class_mode="categorical"
)

resnet_model = create_model(resnet_url, num_classes=train_data_10_percent.num_classes)

resnet_model.compile(
  loss="categorical_crossentropy",
  optimizer=tf.keras.optimizers.Adam(),
  metrics=["accuracy"]
)

resnet_history = resnet_model.fit(
  train_data_10_percent,
  epochs=5,
  steps_per_epoch=len(train_data_10_percent),
  validation_data=test_data,
  validation_steps=len(test_data),
  callbacks=[create_tensorboard_callback(dir_name="tensorflow_hub", experiment_name="ResNet")]
)

# pred_and_plot(resnet_model, "burger.jpg", class_names) # need burger.jpg
resnet_model.evaluate(test_data)
plot_loss(resnet_history)