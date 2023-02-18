#!bin/bash

tensorboard dev upload --logdir ./tensorflow_hub \
  --name "EfficientNet vs. ResNet50V2 vs. MobileNet" \
  --description "Comparing Three different TF Hub feature extraction model architectures using TensorBoard" \
  --one_shot

tensorboard dev list