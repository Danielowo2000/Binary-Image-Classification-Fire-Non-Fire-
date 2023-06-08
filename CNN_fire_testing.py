# Importing the dependencies
import tensorflow as tf
import numpy as np
import cv2

# Reading an image data from the working directory.
the_fire_image = cv2.imread('non_fire.1.png')

# Resizing the image to (128x128x3).
the_fire_image = cv2.resize(the_fire_image, (128, 128))

# Scaling the image pixels between 0 and 1. (255/1-black and 0/0-white).
imagee = np.array(the_fire_image) / 225

# Creating collection of a 3-D image
image = np.expand_dims(imagee, axis=0)

# Loading the best model based on the best validation accuracy after training for 15 epochs (Iterations).
model = tf.keras.models.load_model('fire2.h5')

answer = model.predict(image)
print(answer)
