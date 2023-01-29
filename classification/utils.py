import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import tensorflow as tf
import random

def compare_classification_sets(model, X_train, y_train, X_test, y_test):
  plt.figure(figsize=(12, 6))
  plt.subplot(1, 2, 1)
  plt.title("Train")
  plt_decision_boundary(model, X_train, y_train)
  plt.subplot(1, 2, 2)
  plt.title("Test")
  plt_decision_boundary(model, X_test, y_test)
  plt.show()

def plt_decision_boundary(model, X, y):
  x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
  y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
  xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
  x_in = np.c_[xx.ravel(), yy.ravel()]
  y_pred = model.predict(x_in)

  if len(y_pred[0]) > 1:
    print("doing multiclass classification")
    y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
  else:
    print("doing binary classification")
    y_pred = np.round(y_pred).reshape(xx.shape)

  plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
  plt.scatter(X[:, 0], X[:, 1], c=y, s=5, cmap=plt.cm.RdYlBu)
  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())

def plot_confusion_matrix(y_true, y_pred, figsize, classes=None):
  cm = confusion_matrix(y_true, tf.round(y_pred))
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
  n_classes = cm.shape[0]
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues)
  fig.colorbar(cax)
  labels = None

  if classes:
    labels = classes
  else:
    labels = np.arange(n_classes)

  ax.set(
      title="Confusion Matrix",
      xlabel="Predicted Label",
      ylabel="True Label",
      xticks=np.arange(n_classes),
      yticks=np.arange(n_classes),
      xticklabels=labels,
      yticklabels=labels
  )

  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom();

  ax.xaxis.label.set_size(20)
  ax.yaxis.label.set_size(20)
  ax.title.set_size(20)

  threshold = (cm.max() + cm.min()) / 2.;

  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)", color="white" if cm[i, j] > threshold else "black", size=7, horizontalalignment="center")

def plot_random_image(model, images, y_true, classes):
  i = random.randint(0, len(images))

  target_image = images[i]
  pred_probs = model.predict(target_image.reshape(1, 28, 28))
  pred_class = classes[pred_probs.argmax()]
  true_class = classes[y_true[i]]
  print(true_class)

  plt.imshow(target_image, cmap=plt.cm.Blues)

  if pred_class == true_class:
    color = "green"
  else:
    color = "red"

  plt.xlabel(f"Pred: {pred_class} {100*tf.reduce_max(pred_probs):2.0f}% (True: {true_class})", color=color)