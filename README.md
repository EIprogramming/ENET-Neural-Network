# ENET: A Neural Network from Scratch

  
## Introduction :)
ENET is a fully functional feedforward neural network, made by myself "from scratch" (i.e. without using machine learning libraries like PyTorch or copying any code from others/LLMs) using primarily NumPy and matrix mathematics. **Its performance is comparable and sometimes better than feedforward networks of the same structure implemented in PyTorch and TensorFlow.** During testing, ENET consistently achieved ~98% test accuracy on MNIST digits within 6 epochs (in 22 seconds on my machine).

This was a *highly* enlightening project that I treated above all as an exploratory experience; I watched YouTube videos to get a better understanding, before diving deep into mathematical resources, textbooks, and online resources regarding the math behind backpropagation and neural networks in general.

I have gained a significantly stronger understanding of machine learning, and am excited for what I will explore next! I am currently working on implementing a convolutional layer and integrating it with ENET.

## Features

ENET is capable of the following procedures:
* **Basic Image Classification**
	* Able to identify handwritten digits 0-9 with ~98% test accuracy
	* Able to identify articles of clothing (as per MNIST fashion) with ~87% test accuracy
* **Binary Classification**
	* Able to determine breast cancer with ~89.5% test accuracy given examination measurements
* **General Classification**
	* Able to differentiate between three species of *Iris* with 97% test accuracy given petal and sepal length/width

**Databases tested**: MNIST handwritten digits (98% test accuracy), MNIST fashion, Breast Cancer Wisconsin dataset, Iris

ENET currently has the following features implemented:
* **Fully vectorized, batched forward pass and backpropagation**
	* Capable of fully processing 56000 MNIST training images in 4 seconds total
	* Each training step is batched per a user defined number (e.g. 64 images per training step), and all subsequent operations for calculating the output, gradient, AdamW parameters, etc. are vectorized using NumPy operations
	* Convolutional layer forward pass prototype has a fully vectorized convolution operation I developed myself
* **AdamW Optimization**
	* AdamW provides adaptive learning rates using "Moments" that allows it to dynamically adjust the rate of weight decay during learning
	* The learning rate is able to "refine" itself as the model gets more and more accurate, more efficiently
	* This greatly reduces the need to find the 'perfect' learning rate
	* For more information, see this wonderful resource: https://optimization.cbe.cornell.edu/index.php?title=AdamW
* **Noise Generalization**
* **Dropout**

## Examples

Pictures and examples to be added!

## Resources and Thanks

This project would not have been possible without the dedication and kindness of those willing to teach and share their knowledge with the world. Below are some links to resources I use that have guided me along the path:

3Blue1Brown - Backpropagation Series https://www.youtube.com/watch?v=aircAruvnKk

Sebastian Lague - Programming Adventure: Neural Networks https://www.youtube.com/watch?v=hfMk-kjRv4c

Roger Grosse - UofT Backproagation https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/slides/lec06.pdf

Yufeng Hao et al. - AdamW https://optimization.cbe.cornell.edu/index.php?title=AdamW
