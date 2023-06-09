# Fire and Non-Fire Image Classification using Convolutional Neural Networks (CNN)

This project aims to develop a deep learning model using Convolutional Neural Networks (CNN) to classify images into two categories: fire and non-fire. The model will be trained on a labeled dataset consisting of images containing fires and images without fires. The goal is to accurately predict whether an input image contains a fire or not.

## Dataset
The dataset used for training and evaluation consists of a collection of labeled images. The images are divided into two classes: fire and non-fire. The dataset is split into three subsets: training, validation, and testing. The training set is used to train the model, the validation set is used for hyperparameter tuning and model evaluation during training, and the testing set is used for final evaluation of the trained model.

## Model Architecture
The model architecture employed for this project is a Convolutional Neural Network (CNN). CNNs are well-suited for image classification tasks due to their ability to automatically learn spatial hierarchies and extract meaningful features from images. The architecture consists of multiple convolutional layers, followed by pooling layers to reduce spatial dimensions, and fully connected layers for classification.

The specific architecture and hyperparameters of the CNN model are determined through experimentation and optimization. Techniques such as dropout and batch normalization may be employed to improve the model's generalization and performance.
