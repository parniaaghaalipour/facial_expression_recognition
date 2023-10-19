# facial_expression_recognition

![CD599C75-D27B-46CE-B560-A5D831C02F7A](https://github.com/parniaaghaalipour/facial_expression_recognition/assets/141918224/ad586e15-9cae-4421-a7a2-040ed4e85aed)

This Python program implements a facial expression recognition system. The system is trained using a Convolutional Neural Network (CNN), a class of deep learning algorithms, which are particularly useful for image processing tasks.


## Facial Expression Recognition

Facial Expression Recognition is a field of study in the realm of computer vision and machine learning that focuses on detecting and recognizing human emotions from facial expressions in static images or video sequences. Facial Expression Recognition has a wide range of applications, including but not limited to human-computer interaction, mood-detection software, and medical diagnosis.

## Model

This application constructs a convolutional neural network (CNN) model for the task of facial recognition. The `ConvolutionalNetwork` class represents our model.

The CNN model in this program contains two main layers which are defined in the constructor method. Each layer is a combination of a Convolutional section, Batch Normalization, a ReLU (Rectified Linear Units) activation function, and a Max Pooling layer.

The fully connected layer (`self.fc`) at the end is used to get the final output size matching the number of classes in our dataset which is seven. The seven classes correspond to the seven basic human emotions - Happy, Sad, Surprise, Fear, Anger, Disgust, and Neutral.

## Input

The expected input for our model is a facial image. A class named `DatasetLoader` is utilized to load and preprocess these images. The preprocessing steps include random resizing, random horizontal flipping, converting the image to a tensor, and normalizing the tensor.

The image tensor is then fed into our CNN model. The `DatasetLoader` class uses PyTorch's `ImageFolder` utility to label the images based on the directory structure of the input dataset. The preprocessed and labeled dataset is then loaded into PyTorch's `DataLoader`.

## Output

The output of the model is a tensor of 7 probabilities, each referring to one of the seven human emotional states mentioned above. The emotion with the highest probability is often chosen as the final output of the recognition task.

## Features

The system includes a training function `train_model` that trains the neural network using the Adam optimization algorithm and Cross Entropy Loss function. This function also prints loss information during training for monitoring progress.

The trained model can be saved to disk for future use without needing to retrain it every time.

The major advantage of using a CNN for tasks such as this one is that the model is able to learn and generate its own features during training, typically leading to better performance when compared to traditional machine learning algorithms.

## Future Work

It is possible to further increase the performance of this system by introducing other techniques such as data augmentation, using a more complex model architecture, trying different optimizers and loss function, or increasing the size of the dataset.
