# Transfer Learning

In this section I will tell you what is Transfer Learning and how to use it. **Transfer learning** is a machine learning method where we reuse a pre-trained model as the starting point for a model on a new task. To put it simply—a model trained on one task is repurposed on a second, related task as an optimization that allows rapid progress when modeling the second task.

## Transfer Learning: Feature Extraction

Feature extraction method is a very popular method of training a model which was already pretrained for us before. Below you can find an example of how we can extract features from our model while it is learning.

```python
from tensorflow.keras.layers import GlobalAveragePooling2D

IMG_SIZE = (224, 224)
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
```

Here, to **extract features** from our model I use `GlobalAveragePooling2D` (you can read more about it on the following [page](https://keras.io/api/layers/pooling_layers/global_average_pooling2d/)). This way we squeeze our output data and get the most useful features from it.

## TensorFlow: ModelCheckpoint

We can save our model states while it has been fitting on some data using `ModelCheckpoint` callback. It is often used when our model learns but then starts to outfit and we want to save its best state to return our model to this state after it has finished with fitting.

```python
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
```

Example of fitting a model with `ModelCheckpoint`:

```python
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
```

If we want to load our checkpoint (i.e load our previous state of the model) we can use next method:

```python
model.load_weights(checkpoint_path)
```

## Transfer Learning: Fine-tuning

After we pretrained our model with `feature-extraction`, we can continue our learning for next, for example, 5 epochs and `fine-tune` it. `Fine-tune` means to unfreeze last **n** number of layers of a Neural Network that was previously trained by us with feature extraction method.

In order to do that we need to unfreeze last **n** layer of our model but keep all other layers freezed:

```python
base_model.trainable = True

for layer in base_model.layers[:-10]:
  layer.trainable = False

# Recompile (We have to recompile our models every time we make a change)
model_2.compile(
  loss="categorical_crossentropy",
  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # when fine-tuning, you typically want to lower the learning rate by 10x
  metrics=["accuracy"]
)
```

**Alert!** When we train our model using `fine-tuning` we should lower our `learning_rate` to let our model learn more precisely.
Also, when we unfreeze some of base model layers **we need to recompile model where this base model is used as a layer, we don't need to recompile our base model.**

Here is an example of code how we can fit out model with `fine-tuning`:

```python
initial_epochs = 5
fine_tuning_epochs = initial_epochs + 5

history_fine_10_percent_data_aug = model_2.fit(
    train_data_10_percent,
    epochs=fine_tuning_epochs,
    validation_data=test_data,
    validation_steps=int(0.25 * len(test_data)),
    initial_epoch=model_2_history.epoch[-1], # start training from previous last epoch
    callbacks=[
      create_tensorboard_callback(
        dir_name="transfer_learning",
        experiment_name="10_percent_fine_tune_last_10_layers"
      )
    ]
)
```

We add here only one unknown parameter for us, it calls `initial_epoch`. Here is an explanation from [StackOverflow](https://stackoverflow.com/questions/52476191/what-does-initial-epoch-in-keras-mean) what this parameter means:

Since in some of the **optimizers**, some of their internal values (e.g. **learning rate**) are set using the **current epoch value**, or even you may have (custom) **callbacks** that depend on the current value of epoch, the `initial_epoch` argument let you specify the initial value of `epoch` to start from when training.

As stated in the **documentation**, this is mostly useful when you have trained your model for some epochs, say 10, and then saved it and now you want to load it and resume the training for another 10 epochs **without disrupting the state of epoch-dependent objects (e.g. optimizer)**. So you would set `initial_epoch=10` (i.e. we have trained the model for 10 epochs) and `epochs=20` (not 10, since the total number of epochs to reach is 20) and then everything resume as if you were initially trained the model for 20 epochs in one single training session.

Thus, we can fit our model very good using `feature extraction` and `fine tuning`:

```
Saving TensorBoard log files to: transfer_learning/full_10_classes_fine_tune_last_10/20230218-111821
Epoch 5/10
235/235 [==============================] - 64s 246ms/step - loss: 0.7262 - accuracy: 0.7655 - val_loss: 0.3793 - val_accuracy: 0.8684
Epoch 6/10
235/235 [==============================] - 50s 209ms/step - loss: 0.5931 - accuracy: 0.8080 - val_loss: 0.3443 - val_accuracy: 0.8783
Epoch 7/10
235/235 [==============================] - 45s 189ms/step - loss: 0.5213 - accuracy: 0.8289 - val_loss: 0.3265 - val_accuracy: 0.8816
Epoch 8/10
235/235 [==============================] - 44s 182ms/step - loss: 0.4780 - accuracy: 0.8472 - val_loss: 0.3199 - val_accuracy: 0.8882
Epoch 9/10
235/235 [==============================] - 40s 169ms/step - loss: 0.4395 - accuracy: 0.8564 - val_loss: 0.3069 - val_accuracy: 0.8947
Epoch 10/10
235/235 [==============================] - 40s 169ms/step - loss: 0.4073 - accuracy: 0.8685 - val_loss: 0.3299 - val_accuracy: 0.8882
```

`- val_loss: 0.3299 - val_accuracy: 0.8882`, that is incredible!

## Already pretrained models for Computer Vision problem

- EfficientNet
- ResNet
- MobileNet

These models are used to solve the classification problem.

## Types of Transfer Learning

- **"As is" transfer learning** - using an existing model with no changes what so ever (e.g ImageNet model on 1000 ImageNet classes, none of your own)
- **"Feature extraction"** transfer learning - use the prelearned patterns of an existing model (e.g EfficientNetB0 trained on ImageNet) and adjust the output layer for your own problem (e.g 1000 classes -> 10 classes of food)
- **"Fine-tuning"** transfer learning - use the prelearned patterns of an existing model and "fine-tune" many or all of the underlying layers (including new output layers)

## Sequential API vs. Functional API

```python
# Sequential API

sequential_model = tf.keras.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation="relu"),
  tf.keras.layers.Dense(64, activation="relu"),
  tf.keras.layers.Dense(10, activation="softmax")
], name="sequential_model")
```

```python
# Functional API

inputs = tf.keras.layers.Input(shape=(28, 28))
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
functional_model = tf.keras.Model(inputs, outputs, name="functional_model")
```

## Other Terms

- [BatchDataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)
- [TensorBoard callback](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/TensorBoard)
- [ImageNet database](https://www.image-net.org/)
- [ModelCheckpoint](https://keras.io/api/callbacks/model_checkpoint/)
- [TensorFlow Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)
