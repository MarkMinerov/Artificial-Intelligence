# How to build build a regression model

- Get data in DataFrames
- Scale data using next methods:

```python
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

ct = make_column_transformer(
  (MinMaxScaler(), ["columns", "we", "want to", "scale"]),
  (OneHotEncoder(handle_unknown="ignore"), ["One Hot encode"])
)

X = insurance.drop("drop Y", axis=1)
Y = insurance["Y"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

ct.fit(X_train)

X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)
```

- Create Sequential model using:

```python
insurance_model = tf.keras.Sequential([
  # layers...
])
```

- it is assumed that the layers go from high to low:

```python
insurance_model = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(80),
    tf.keras.layers.Dense(50),
    tf.keras.layers.Dense(20),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])
```

Here I have layers which amount of neurons lower: 100, 80, 50, 20, 10, 1...

- Add Dropout if needed:

_In the overfitting problem, the model learns the statistical noise. To be precise, the main motive of training is to decrease the loss function, given all the units (neurons). So in overfitting, a unit may change in a way that fixes up the mistakes of the other units. This leads to complex co-adaptations, which in turn leads to the overfitting problem because this complex co-adaptation fails to generalise on the unseen dataset._

_Now, if we use dropout, it prevents these units to fix up the mistake of other units, thus preventing co-adaptation, as in every iteration the presence of a unit is highly unreliable. So by randomly dropping a few units (nodes), it forces the layers to take more or less responsibility for the input by taking a probabilistic approach._

_This ensures that the model is getting generalised and hence reducing the overfitting problem._

- Compile your model using

```python
model.compile(loss=loss_function, optimizer=optimizer, metrics=array_of_metrics)
```

1. [**Loss Functions**](https://keras.io/api/losses/)
2. [**Optimizers**](https://keras.io/api/optimizers/)
3. [**Metrics**](https://keras.io/api/metrics/)

- Evaluate your model using `model.evaluate` method passing `test_X` and `test_Y` as parameters

- Tweak your model if needed in order to make your model predict more precisely:

1. Change amount of layers
2. Change units in layers
3. Add activation function to layers **if needed**
4. Change optimizer (the most used: SGD, Adam)
5. Change Loss function **if needed**
6. Add more data for learning if possible
