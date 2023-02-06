from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)

train_datagen_augmented = ImageDataGenerator(
  rescale=1/255.,
  rotation_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  width_shift_range=0.2,
  height_shift_range=0.2,
  horizontal_flip=True
)

test_datagen = ImageDataGenerator(
  rescale=1/255.
)

train_data_augmented = train_datagen_augmented.flow_from_directory(
  directory="pizza_steak/train",
  target_size=IMG_SIZE,
  class_mode="binary",
)

test_data = test_datagen.flow_from_directory(
  directory="pizza_steak/test",
  target_size=IMG_SIZE,
  class_mode="binary",
)

model = Sequential([
  Conv2D(10, 3, activation="relu", input_shape=(224, 224, 3)),
  MaxPool2D(),
  Conv2D(10, 3, activation="relu"),
  MaxPool2D(),
  Conv2D(10, 3, activation="relu"),
  MaxPool2D(),
  Flatten(),
  Dense(1, activation="sigmoid"),
])

model.compile(loss="binary_crossentropy", optimizer=Adam(), metrics="accuracy")

history_model_4 = model.fit(
  train_data_augmented,
  epochs=5,
  steps_per_epoch=len(train_data_augmented),
  validation_data=test_data,
  validation_steps=len(test_data)
)