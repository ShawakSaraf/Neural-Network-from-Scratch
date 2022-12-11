# Building a Neural Network from scratch

You know the rite of passage everyone has to go through when they first step foot into the world of programming?  
The iconic, print("Hello World!").

This is that but for the world of Neural Networks. An implementation of a basic neural network built solely using NumPy library. This was a small project I started when I first started doing research about neural networks and found the absolutly incredible book, ["Neural Networks and Deep Learning"](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen. They did such an amazing job of explaining the magic behind these nerworks with code examples, I used it as a refrence to do my own implementation.

This dense, fully connected neural network — a more technical term would be, a Multi-Layer Perceptron (MLP) — is capable of performing basic tasks such as image classification, as shown below, and can serve as a good starting point for more advanced deep learning projects.
 
https://user-images.githubusercontent.com/74816223/206852868-7cf65399-7504-4640-910c-1b9d97597dab.mp4  

A MLP is a type of feedforward artificial neural network. It is composed of multiple layers of nodes, where each layer is fully connected to the next. In other words, each node in one layer has a weighted connection to every node in the next layer. MLPs are commonly used for supervised learning tasks, such as classification and regression.

I used the famous MNIST dataset, which consists of 70,000 images of handwritten digits (0-9) and their corresponding labels for training. This dataset is commonly used as a benchmark for evaluating the performance of different machine learning algorithms.

By implementing an MLP using only the Numpy library, I was able to understand the core principles and methametics of how a neural network works. Numpy is a powerful library for numerical computing in Python, but it does not provide high-level abstractions for building and training neural networks. As a result, I had a lot of fun implementing the core components of an MLP from scratch.

We can boil down any neural network into two algorithms, forward and backward propagation.

In the forward propagation step, the input data is passed through the network, layer by layer, and the predicted output is produced. This is done by taking the dot product of the input data and the weights of the connections between the input and the first hidden layer, and then applying an activation function to the result. This process is repeated for each subsequent layer, until the final output is produced.

In the backward propagation step, the error between the predicted output and the true label is calculated, and this error is then used to update the weights of the connections in the network in a way that reduces the error. This is done through the calculation of gradients, which indicate the direction in which the weights should be adjusted in order to minimize the error.

This project turned out to be a great exercise to learn the fundamental concepts and principles of neural networks and deep learning. I was able to gain hands-on experience in building and training a neural network without the bells and whistles of state of the art frameworks like Tensorflow and Pytorch.

