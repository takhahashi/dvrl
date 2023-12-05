import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

model = keras.Sequential(
    [
        layers.Dense(2, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, activation="sigmoid",name="layer3"),
    ]
)
pred_model = keras.models.Sequential()
pred_model.add(keras.layers.Dense(3, activation='relu'))
pred_model.add(keras.layers.Dense(3, activation='relu'))
pred_model.add(keras.layers.Dense(1, activation="sigmoid"))
pred_model.compile(optimizer='adam', loss='mean_squared_error',
                   metrics=['mae'])
# Call model on a test input
x = tf.ones((3, 3))
print(x)
y = pred_model(x)
print(y)

x_train = pd.read_csv("test.csv").values
print(type(x_train))