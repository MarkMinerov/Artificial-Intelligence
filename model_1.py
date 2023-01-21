import tensorflow as tf
from tensorflow.keras.utils import plot_model

from shared.plot_predictions import plot_predictions
from shared.R_squared import R_squared

X = tf.range(-100, 500, 2)
Y = X + 5

X_train = X[:230]
Y_train = Y[:230]

X_test = X[230:]
Y_test = Y[230:]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(50),
    tf.keras.layers.Dense(1),
 ])

model.compile(loss=tf.keras.losses.mse, optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=["mse"])

model.fit(tf.expand_dims(X_train, axis=-1), Y_train, epochs=150, verbose=0)
plot_model(model=model, show_shapes=True)

Y_pred = model.predict(X_test)
plot_predictions(predictions=Y_pred, train_data=X_train, train_labels=Y_train, test_data=X_test, test_labels=Y_test)

MAE_error = tf.keras.metrics.mean_absolute_error(Y_test, tf.squeeze(Y_pred))
MSE_error = tf.keras.metrics.mean_squared_error(Y_test, tf.squeeze(Y_pred))

print(MAE_error)
print(MSE_error)

print(R_squared(Y_test, tf.squeeze(Y_pred)).numpy())
model.save("saved/model_1")
model.save("saved/model_1.h5")