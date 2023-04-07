# How to train and evaluate our model on Google Cloud

Here I describe what commands we need to use in order to allow our model start training on Google Cloud since it is better and faster to train models on it.

## Google SDK command to begin training process

```sh
export MODEL_DIR=[PATH TO training_process]
export PIPELINE_CONFIG_PATH=[PATH TO .config FILE]
```

```sh
# From tensorflow/models/research
cp object_detection/packages/tf2/setup.py .

gcloud ai-platform jobs submit training [JOB NAME] \
  --runtime-version 2.11 \
  --python-version 3.7 \
  --job-dir=gs://${MODEL_DIR} \
  --package-path ./object_detection \
  --module-name object_detection.model_main_tf2 \
  --region us-central1 \
  --scale-tier CUSTOM \
  --master-machine-type n1-highcpu-16 \
  --master-accelerator count=2,type=nvidia-tesla-v100 \
  -- \
  --model_dir=gs://${MODEL_DIR} \
  --pipeline_config_path=gs://${PIPELINE_CONFIG_PATH}
```

## Google SDK command to begin evaluating process

From the tensorflow/models/research/ directory

```sh
cp object_detection/packages/tf2/setup.py .
```

```sh
export MODEL_DIR=[PATH TO training_process]
export PIPELINE_CONFIG_PATH=[PATH TO .config FILE]

gcloud ai-platform jobs submit training [JOB NAME] \
  --runtime-version 2.1 \
  --python-version 3.7 \
  --job-dir=gs://${MODEL_DIR} \
  --package-path ./object_detection \
  --module-name object_detection.model_main_tf2 \
  --region us-central1 \
  --scale-tier BASIC_GPU \
  -- \
  --model_dir=gs://${MODEL_DIR} \
  --pipeline_config_path=gs://${PIPELINE_CONFIG_PATH} \
  --checkpoint_dir=gs://${MODEL_DIR}
```

[Job submission documentation](https://cloud.google.com/sdk/gcloud/reference/ai-platform/jobs/submit)

```sh
--runtime-version=RUNTIME_VERSION
```

AI Platform runtime version for this job. Must be specified unless --master-image-uri is specified instead.
It is defined in documentation along with the list of supported versions:
https://cloud.google.com/ai-platform/prediction/docs/runtime-version-list

```sh
--job-dir=JOB_DIR
```

Cloud Storage path in which to store training outputs and other data needed for training.
This path will be passed to your TensorFlow program as the --job-dir command-line arg.
The benefit of specifying this field is that AI Platform will validate the path for use
in training. However, note that your training program will need to parse the provided --job-dir argument.

```sh
--package-path=PACKAGE_PATH
```

Path to a Python package to build. This should point to a local directory containing the Python source for the job.
It will be built using setuptools (which must be installed) using its parent directory as context.
If the parent directory contains a setup.py file, the build will use that; otherwise, it will use a simple built-in one.

```sh
--module-name=MODULE_NAME
```

Name of the module to run.

```sh
--region=REGION
```

Region of the machine learning training job to submit.

```sh
--scale-tier=SCALE_TIER
```

Specify the machine types, the number of replicas for workers, and parameter servers. SCALE_TIER must be one of:

- `basic` - Single worker instance. This tier is suitable for learning how to use AI Platform, and for experimenting with new models using small datasets.
- `basic-gpu` - Single worker instance with a GPU.
- `basic-tpu` - Single worker instance with a Cloud TPU.
- `custom` - You can read more about this parameter on the following [page](https://cloud.google.com/sdk/gcloud/reference/ai-platform/jobs/submit/training#--scale-tier)

```sh
--model_dir=gs://${MODEL_DIR}
```

Where to save our progress while training

```sh
--pipeline_config_path=gs://${PIPELINE_CONFIG_PATH}
```

Path to our config file

## Train model using Google SDK Docker API

```sh
gcloud ai-platform jobs submit training [JOB NAME] \
  --job-dir=gs://${MODEL_DIR} \
  --region us-central1 \
  --scale-tier CUSTOM \
  --master-machine-type n1-highcpu-16 \
  --master-accelerator count=8,type=nvidia-tesla-v100 \
  --master-image-uri gcr.io/${DOCKER_IMAGE_URI} \
  -- \
  --model_dir=gs://${MODEL_DIR} \
  --pipeline_config_path=gs://${PIPELINE_CONFIG_PATH}
```

Documentation of used parameters

```sh
--master-machine-type=MASTER_MACHINE_TYPE
```

Specifies the type of virtual machine to use for training job's master worker.
You must set this value when `--scale-tier` is set to `CUSTOM`.

```sh
--master-accelerator=[count=COUNT],[type=TYPE]
```

Hardware accelerator config for the master worker. Must specify both the accelerator type `(TYPE)` for each server and the number of accelerators to attach to each server `(COUNT)`.

- type - Type of the accelerator. Choices are `nvidia-tesla-a100`,`nvidia-tesla-k80`,`nvidia-tesla-p100`,`nvidia-tesla-p4`,`nvidia-tesla-t4`,`nvidia-tesla-v100`,`tpu-v2`,`tpu-v2-pod`,`tpu-v3`,`tpu-v3-pod`

- count - Number of accelerators to attach to each machine running the job. Must be greater than 0.

```sh
--master-image-uri=MASTER_IMAGE_URI
```

Docker image to run on each master worker. This image must be in Container Registry. Only one of `--master-image-uri` and `--runtime-version` must be specified.
