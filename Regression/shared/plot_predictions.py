import matplotlib.pyplot as plt

def plot_predictions(train_data, train_labels, test_data, test_labels, predictions):
  plt.figure(figsize=(10, 7))
  plt.scatter(train_data, train_labels, c="b", label="Training data")
  plt.scatter(test_data, test_labels, c="g", label="Testing data")
  plt.scatter(test_data, predictions, c="r", label="Predictions")
  plt.legend()
  plt.show()