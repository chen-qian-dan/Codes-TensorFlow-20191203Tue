
# import libraries
import tensorflow as tf
from tensorflow import keras # API of tensorflow
import numpy as np
import matplotlib.pyplot as plt

# load data------------------------------------------
data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# scale images
train_images = train_images / 255.0
test_images = test_images / 255.0

print(train_images[7])
plt.imshow(train_images[7], cmap=plt.cm.binary)
plt.show()

print(train_labels[0])