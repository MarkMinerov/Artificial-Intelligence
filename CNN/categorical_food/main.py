import pathlib
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation
from tensorflow.keras import Sequential

train_dir = "10_food_classes_all_data/train/"
test_dir = "10_food_classes_all_data/test/"

data_dir = pathlib.Path(train_dir)
class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen_augmented = ImageDataGenerator(
  rescale=1/255.,
  rotation_range=0.2,
  width_shift_range=0.2,
  height_shift_range=0.2,
  zoom_range=0.2
)

test_datagen = ImageDataGenerator(rescale=1/255.)

train_data = train_datagen_augmented.flow_from_directory(train_dir, target_size=(224, 224))
test_data = test_datagen.flow_from_directory(test_dir, target_size=(224, 224))

model_1 = Sequential([
    Conv2D(10, 3, input_shape=(224, 224, 3)),
    Activation("relu"),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(),
    Conv2D(10, 3, activation="relu"),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(),
    Flatten(),
    Dense(10, activation="softmax"),
])

model_1.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])

model_1_history = model_1.fit(
  train_data,
  epochs=10,
  steps_per_epoch=len(train_data),
  validation_data=test_data,
  validation_steps=len(test_data)
)

model_1.save('model_1.h5')