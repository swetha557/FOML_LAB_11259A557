
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Load Fashion-MNIST dataset
fashion = fetch_openml("Fashion-MNIST", version=1, as_frame=False)

# Normalize pixel values
X = fashion.data.astype("float32") / 255.0

# Convert target labels into integers
y = fashion.target.astype("int")

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# Build the Artificial Neural Network
model = tf.keras.Sequential([
    tf.keras.Input(shape=(784,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

# Display accuracy
print("Test accuracy:", round(accuracy, 3))

new_image = X_test[0:1]                           # FIX: X_test[0:1] is a proper 2-D array with 1 row
prediction = model.predict(new_image, verbose=0)  # one probability per class
item = np.argmax(prediction)                      # pick the most likely class number
clothes = ["T-shirt","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]  # names for classes 0 to 9
print("Predicted item:", clothes[item])           # turn class number into a name