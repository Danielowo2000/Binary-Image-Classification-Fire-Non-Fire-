# Importing our dependencies
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

im_width = 128
im_height = 128

path = "fire_dataset_test"
path_v = "fire_dataset_train"

# Rescaling the image pixels for train set.
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    # shear_range=0.2,
    # zoom_range=0.2,
    # No change in image orientation.
    horizontal_flip=False)

# Rescaling the image pixels for test set.
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    path,  # this is the target directory
    target_size=(im_width, im_width),  # all images will be resized to 128x128
    # Each batch of 64 images will contain a mixture of both classes
    class_mode='binary',
    color_mode="rgb",
    batch_size=64
)

validation_generator = test_datagen.flow_from_directory(
    path_v,
    target_size=(im_width, im_width),
    class_mode='binary',
    color_mode="rgb",
    batch_size=64
)

# Convolutional Neural Network Architecture.
model = keras.models.Sequential()
# Applying 16 filters to each input to get 16 output feature maps
model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=(128, 128, 3)))
# Applying max pooling to disregard less relevant features and extracting important features in the image.
model.add(MaxPooling2D(2, 2))
# Applying 8 filters to each input to get 8 output feature maps
model.add(Conv2D(filters=8, kernel_size=(3, 3), padding='valid', activation='relu'))
model.add(MaxPooling2D(2, 2))
# Stretching the image for the fully connected network
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(5, activation='relu'))
# Binary classification uses one Neuron at the output.
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()
callbacks = [EarlyStopping(patience=5, verbose=1),
             ReduceLROnPlateau(factor=0.15, patience=3, min_Ir=0.00001, verbose=1),
             ModelCheckpoint('fire2.h5', verbose=1, save_best_only=True, save_weights_only=False)]

# Validating the data after each epoch for monitoring (To prevent overfitting)
model.fit(train_generator, validation_data=validation_generator, batch_size=64, epochs=15, callbacks=callbacks)
