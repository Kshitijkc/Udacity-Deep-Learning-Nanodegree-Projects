# Udacity Deep Learning Nanodegree

This repository contains material related to Udacity's [Deep Learning Nanodegree Foundation](https://www.udacity.com/course/deep-learning-nanodegree-foundation--nd101) program. It consists of a bunch of tutorial notebooks for various deep learning topics. In most cases, the notebooks lead you through implementing models such as convolutional networks, recurrent networks, and GANs. There are other topics covered such as weight intialization and batch normalization.

There are also notebooks used as projects for the Nanodegree program. In the program itself, the projects are reviewed by Udacity experts, but they are available here as well.

## Table Of Contents


### Projects

* [Your First Neural Network](https://github.com/Kshitijkc/Udacity-Deep-Learning-Nanodegree-Projects/tree/master/1. Predicting Bike Sharing Data): Implement a neural network in Numpy to predict bike rentals.
* [Image classification](https://github.com/udacity/deep-learning/tree/master/image-classification): Build a convolutional neural network with TensorFlow to classify CIFAR-10 images.
* [Text Generation](https://github.com/udacity/deep-learning/tree/master/tv-script-generation): Train a recurrent neural network on scripts from The Simpson's (copyright Fox) to generate new scripts.
* [Machine Translation](https://github.com/udacity/deep-learning/tree/master/language-translation): Train a sequence to sequence network for English to French translation (on a simple dataset)
* [Face Generation](https://github.com/udacity/deep-learning/tree/master/face_generation): Use a DCGAN on the CelebA dataset to generate images of novel and realistic human faces.


## Dependencies

Each directory has a `requirements.txt` describing the minimal dependencies required to run the notebooks in that directory.

### pip

To install these dependencies with pip, you can issue `pip3 install -r requirements.txt`.

### Conda Environments

You can find Conda environment files for the Deep Learning program in the `environments` folder. Note that environment files are platform dependent. Versions with `tensorflow-gpu` are labeled in the filename with "GPU".
